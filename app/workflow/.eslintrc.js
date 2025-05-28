const { defineConfig } = require('@flowgram.ai/eslint-config');

module.exports = defineConfig({
  preset: 'web',
  packageRoot: __dirname,
  rules: {
    'no-console': 'off',
    'react/prop-types': 'off',
  },
  settings: {
    react: {
      version: 'detect', // 自动检测 React 版本
    },
  },
});
