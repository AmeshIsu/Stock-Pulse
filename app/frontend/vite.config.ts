import { defineConfig } from 'vitest/config'
import react from '@vitejs/plugin-react'

// https://vite.dev/config/
export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    proxy: {
      '/api': {
        target: 'http://127.0.0.1:8000',
        changeOrigin: true,
        secure: false,
      },
    },
  },
  build: {
    rollupOptions: {
      output: {
        manualChunks(id) {
          // All admin routes and components are grouped into chunks prefixed with admin-
          if (id.includes('/admin/') || id.includes('\\admin\\') || id.includes('pages/admin') || id.includes('pages\\admin')) {
            const parts = id.split(/[/\\]/);
            const idx = parts.findIndex(p => p === 'admin');
            const sub = idx >= 0 && parts[idx + 1] ? parts[idx + 1].replace(/\.[^/.]+$/, '') : 'core';
            return `admin-${sub}`;
          }
        },
      },
    },
  },
  test: {
    globals: true,
    environment: 'jsdom',
    setupFiles: './src/test/setup.ts',
    css: true,
    include: ['src/**/*.{test,spec}.{ts,tsx}'],
  },
})
