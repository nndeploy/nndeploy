import { pluginReact } from '@rsbuild/plugin-react';
import { pluginLess } from '@rsbuild/plugin-less';
import { defineConfig } from '@rsbuild/core';

export default defineConfig({
  plugins: [pluginReact(), pluginLess()],
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
    title: 'demo-free-layout',
  },
});
