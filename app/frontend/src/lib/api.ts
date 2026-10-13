// API Client for Stock Pulse with automatic CSRF header and 401 refresh flow

const API_BASE_URL = import.meta.env.VITE_API_BASE_URL || '/api/v1';

function getCookie(name: string): string | null {
  const match = document.cookie.match(new RegExp('(^|;\\s*)(' + name + ')=([^;]*)'));
  return match ? decodeURIComponent(match[3]) : null;
}

export interface ApiErrorResponse {
  error: {
    code: string;
    message: string;
    details?: unknown;
  };
}

export class ApiError extends Error {
  code: string;
  status: number;
  details?: unknown;

  constructor(status: number, code: string, message: string, details?: unknown) {
    super(message);
    this.name = 'ApiError';
    this.status = status;
    this.code = code;
    this.details = details;
  }
}

let isRefreshing = false;
let refreshSubscribers: ((token: boolean) => void)[] = [];

function subscribeTokenRefresh(cb: (token: boolean) => void) {
  refreshSubscribers.push(cb);
}

function onRefreshed(success: boolean) {
  refreshSubscribers.forEach((cb) => cb(success));
  refreshSubscribers = [];
}

export async function apiClient<T>(
  endpoint: string,
  options: RequestInit = {}
): Promise<T> {
  const url = endpoint.startsWith('http')
    ? endpoint
    : `${API_BASE_URL}${endpoint.startsWith('/') ? endpoint : `/${endpoint}`}`;

  const headers = new Headers(options.headers || {});
  if (!headers.has('Content-Type') && !(options.body instanceof FormData)) {
    headers.set('Content-Type', 'application/json');
  }

  // Include CSRF token on state-modifying requests
  const method = (options.method || 'GET').toUpperCase();
  if (['POST', 'PUT', 'DELETE', 'PATCH'].includes(method)) {
    const csrfToken = getCookie('csrf_token') || getCookie('stockpulse_csrf');
    if (csrfToken) {
      headers.set('X-CSRF-Token', csrfToken);
    }
  }

  const config: RequestInit = {
    ...options,
    headers,
    credentials: 'include', // Ensure cookies are sent and received
  };

  let response: Response;
  try {
    response = await fetch(url, config);
  } catch (err: unknown) {
    throw new ApiError(0, 'NETWORK_ERROR', err instanceof Error ? err.message : 'Network connection failed');
  }

  // Handle 401 Unauthorized with token refresh attempt (once)
  if (response.status === 401 && !endpoint.includes('/auth/login') && !endpoint.includes('/auth/refresh') && !endpoint.includes('/public/info')) {
    if (!isRefreshing) {
      isRefreshing = true;
      try {
        const refreshRes = await fetch(`${API_BASE_URL}/auth/refresh`, {
          method: 'POST',
          credentials: 'include',
        });
        if (refreshRes.ok) {
          isRefreshing = false;
          onRefreshed(true);
        } else {
          isRefreshing = false;
          onRefreshed(false);
          // Refresh failed, propagate 401
          throw new ApiError(401, 'UNAUTHORIZED', 'Session expired. Please log in again.');
        }
      } catch {
        isRefreshing = false;
        onRefreshed(false);
        throw new ApiError(401, 'UNAUTHORIZED', 'Session expired. Please log in again.');
      }
    } else {
      // Wait for ongoing refresh
      const refreshed = await new Promise<boolean>((resolve) => subscribeTokenRefresh(resolve));
      if (!refreshed) {
        throw new ApiError(401, 'UNAUTHORIZED', 'Session expired. Please log in again.');
      }
    }

    // Retry original request once
    if (['POST', 'PUT', 'DELETE', 'PATCH'].includes(method)) {
      const csrfToken = getCookie('stockpulse_csrf');
      if (csrfToken) {
        headers.set('X-CSRF-Token', csrfToken);
      }
    }
    response = await fetch(url, { ...config, headers });
  }

  // Parse JSON or text
  const contentType = response.headers.get('content-type') || '';
  if (response.status === 204) {
    return {} as T;
  }

  if (contentType.includes('application/json')) {
    const data = await response.json();
    if (!response.ok) {
      const err = data as ApiErrorResponse;
      throw new ApiError(
        response.status,
        err.error?.code || 'UNKNOWN_ERROR',
        err.error?.message || `HTTP error ${response.status}`,
        err.error?.details
      );
    }
    return data as T;
  }

  if (!response.ok) {
    const text = await response.text();
    throw new ApiError(response.status, 'HTTP_ERROR', text || `HTTP error ${response.status}`);
  }

  return (await response.text()) as unknown as T;
}
