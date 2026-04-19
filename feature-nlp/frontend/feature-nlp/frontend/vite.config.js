import { defineConfig, loadEnv } from 'vite'
import react from '@vitejs/plugin-react'

export default defineConfig(({ mode }) => {
  const env = loadEnv(mode, process.cwd(), '')
  const proxyTarget = env.VITE_PROXY_TARGET || 'http://localhost:8000'

  return {
    plugins: [react()],
    server: {
      port: 5173,
      proxy: {
        '/upload': proxyTarget,
        '/query': proxyTarget,
        '/health': proxyTarget,
        '/data-health': proxyTarget,
        '/auto-visualize': proxyTarget,
        '/correlation-matrix': proxyTarget,
        '/generate-sql': proxyTarget,
      },
    },
  }
})
