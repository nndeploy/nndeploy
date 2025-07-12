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
    title: 'demo-free-layout',
  },
  server: {
    port: 3000, // 开发服务器的端口
    proxy: {

      '/api/ws': { // 代理路径，可以是任意路径
        target: 'ws://183.215.186.194:9999', // 目标服务器的地址
         ws: true, 
        secure: false, // 如果目标服务器使用自签名证书，请设置为false
        changeOrigin: true, // 是否改变请求头中的origin
       
       
       // pathRewrite: { '^/api': '' }, // 重写路径，例如将'/api/user'重写为'/user'
        headers: { // 可选的，向目标服务器发送额外的请求头
          Host: 'localhost'
        }
      }, 
      '/api': { // 代理路径，可以是任意路径
        target: 'http://183.215.186.194:9999', // 目标服务器的地址
        secure: false, // 如果目标服务器使用自签名证书，请设置为false
        changeOrigin: true, // 是否改变请求头中的origin
       // pathRewrite: { '^/api': '' }, // 重写路径，例如将'/api/user'重写为'/user'
        headers: { // 可选的，向目标服务器发送额外的请求头
          Host: 'localhost'
        }
      }

    }
  }
});
