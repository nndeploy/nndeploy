import { pluginReact } from '@rsbuild/plugin-react';
import { pluginLess } from '@rsbuild/plugin-less';
import { pluginSass } from '@rsbuild/plugin-sass';
import { defineConfig } from '@rsbuild/core';

export default defineConfig({
  plugins: [pluginReact(), pluginLess(), pluginSass()],
  source: {
    entry: {
      index: './src/app.tsx',
    },
    /**
     * support inversify @injectable() and @inject decorators
     */
    decorators: {
      version: 'legacy',
    },
  },
  html: {
    title: 'nndeploy',
  },
  server: {
    port: 3000, // 开发服务器的端口
    proxy: {
      '/api': {
        target: process.env.NNDEPLOY_SERVER_URL || 'http://127.0.0.1:8000',
        ws: true,
        // Keep the browser origin and Host consistent for session/CSRF checks.
        changeOrigin: false,
      },
      '/v1': {
        target: process.env.NNDEPLOY_SERVER_URL || 'http://127.0.0.1:8000',
        changeOrigin: false,
      },
    },
  },
});
