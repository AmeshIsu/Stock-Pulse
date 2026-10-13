import { describe, it, expect, vi, beforeEach } from 'vitest';
import { apiClient, ApiError } from './api';

describe('apiClient', () => {
  beforeEach(() => {
    vi.restoreAllMocks();
    document.cookie = '';
  });

  it('injects X-CSRF-Token header on POST requests when cookie is present', async () => {
    document.cookie = 'stockpulse_csrf=test-csrf-token-123';

    let capturedHeaders: Headers | null = null;
    vi.spyOn(global, 'fetch').mockImplementation(async (input, init) => {
      capturedHeaders = new Headers(init?.headers);
      return new Response(JSON.stringify({ status: 'ok' }), {
        status: 200,
        headers: { 'Content-Type': 'application/json' },
      });
    });

    const res = await apiClient<{ status: string }>('/test', {
      method: 'POST',
      body: JSON.stringify({ hello: 'world' }),
    });

    expect(res.status).toBe('ok');
    expect(capturedHeaders?.get('X-CSRF-Token')).toBe('test-csrf-token-123');
  });

  it('surfaces backend error code and message on 400 Bad Request', async () => {
    vi.spyOn(global, 'fetch').mockImplementation(async () => {
      return new Response(
        JSON.stringify({
          error: {
            code: 'VALIDATION_FAILED',
            message: 'Invalid input parameters provided',
          },
        }),
        {
          status: 400,
          headers: { 'Content-Type': 'application/json' },
        }
      );
    });

    await expect(apiClient('/test')).rejects.toThrow(
      'Invalid input parameters provided'
    );
  });

  it('attempts token refresh on 401 Unauthorized', async () => {
    let callCount = 0;
    vi.spyOn(global, 'fetch').mockImplementation(async (url) => {
      callCount++;
      const urlStr = url.toString();
      if (urlStr.includes('/auth/refresh')) {
        return new Response(JSON.stringify({ status: 'refreshed' }), {
          status: 200,
          headers: { 'Content-Type': 'application/json' },
        });
      }
      if (callCount === 1) {
        return new Response(JSON.stringify({ error: { code: 'UNAUTHORIZED', message: 'Token expired' } }), {
          status: 401,
          headers: { 'Content-Type': 'application/json' },
        });
      }
      return new Response(JSON.stringify({ data: 'success' }), {
        status: 200,
        headers: { 'Content-Type': 'application/json' },
      });
    });

    const res = await apiClient<{ data: string }>('/protected-resource');
    expect(res.data).toBe('success');
    expect(callCount).toBe(3); // Initial 401, refresh, retry
  });
});
