C++ API 参考
============

.. note::
   C++ API 文档需要手动生成。

如何构建和查看文档
------------------

1. 下载源码::

   git clone https://github.com/nndeploy/nndeploy.git

   cd nndeploy

2. 构建 C++ API 文档::

   cd docs/zh_cn

   doxygen

3. 启动本地服务器查看文档::

   cd build_doxygen/html
   
   python -m http.server 8000

4. 在浏览器中打开 `http://localhost:8000` 查看生成的文档
