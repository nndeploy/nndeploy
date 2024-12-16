# 开发调试问题记录

## Python

python binding相关，移步`python/README.md`

## 使用safetensors加载的权重修改报错

safetensors方式加载的权重直接mmap到文件，以只读方式打开，无法修改。如果要修改权重，clone一份Tensor修改，并替换。

## Ascend开发环境

### 编译报错，链接错误，undefined reference

需要使用root方式编译

### Vscode终端无路径提示、部分常用命令错误、无法输入等

Vscode默认打开的终端有残缺，需要手动新建终端，选择Bash

### cmake方式
+ 具体步骤
  + 在根目录创建`build`目录，将`cmake/config.cmake`复制到该目录
    ```
    mkdir build
    cp cmake/config.cmake build
    cd build
    ```

  + 编辑`build/config.cmake`自定义编译选项（笔者的自定义编译选项：[path/cmake/config_ascendcl.cmake](../../../cmake/config_ascendcl.cmake)）
      
  + `cmake`
    ```
    cmake ../ -DCMAKE_CXX_COMPILER=g++ -DCMAKE_SKIP_RPATH=TRUE
    ```

### 在开发服务器上，模型以及三方库资源放在`/home/resource`，所有可以登录该服务器的同学都有权限访问该目录

+ 模型资源：`/home/resource/model_zoo`

+ 三方库资源：`/home/resource/third_party`

  + 本地终端起效：
    + export LD_LIBRARY_PATH=/home/ascenduserdg01/github/nndeploy/build:$LD_LIBRARY_PATH
    + export LD_LIBRARY_PATH=/home/resource/third_party/onnxruntime-linux-aarch64-1.20.1/lib:$LD_LIBRARY_PATH
  + 始终有效
    + echo 'export LD_LIBRARY_PATH=/home/ascenduserdg01/github/nndeploy/build:$LD_LIBRARY_PATH' >> ~/.bashrc
    + echo 'export LD_LIBRARY_PATH=/home/resource/third_party/onnxruntime-linux-aarch64-1.20.1/lib:$LD_LIBRARY_PATH' >> ~/.bashrc
    + source ~/.bashrc

+ 图片资源：`/home/resource/data_set`

+ 该开发服务器下，所有开发者具备该目录权限，具体执行命令如下：
  ```shell
  sudo chown -R :nndeploy /home/resource/
  sudo chmod -R g+rwx /home/resource/
  ```