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

### 关于第三方 tokenizer 在编译时产生的 protobuf 冲突问题

#### 问题描述

直接编译后，执行相关 demo 的可执行文件会发生 segment fault，gdb 查看结果如下

```bash
raymond@ascenduserdg01:~/workspace/nndeploy/build$ gdb ./nndeploy_demo_dag
...
Thread 1 "nndeploy_demo_d" received signal SIGSEGV, Segmentation fault.
0x0000ffffecb9c744 in google::protobuf::Arena::AllocateAlignedNoHook(unsigned long) () from /lib/aarch64-linux-gnu/libprotobuf.so.23
(gdb) bt
#0  0x0000ffffecb9c744 in google::protobuf::Arena::AllocateAlignedNoHook(unsigned long) () from /lib/aarch64-linux-gnu/libprotobuf.so.23
#1  0x0000ffffecc50d08 in google::protobuf::FileDescriptorProto* google::protobuf::Arena::CreateMaybeMessage<google::protobuf::FileDescriptorProto>(google::protobuf::Arena*) ()
   from /lib/aarch64-linux-gnu/libprotobuf.so.23
#2  0x0000ffffecc25d70 in google::protobuf::EncodedDescriptorDatabase::Add(void const*, int) () from /lib/aarch64-linux-gnu/libprotobuf.so.23
#3  0x0000ffffecbf8ac8 in google::protobuf::DescriptorPool::InternalAddGeneratedFile(void const*, int) () from /lib/aarch64-linux-gnu/libprotobuf.so.23
#4  0x0000ffffecc6aaa0 in google::protobuf::internal::AddDescriptors(google::protobuf::internal::DescriptorTable const*) () from /lib/aarch64-linux-gnu/libprotobuf.so.23
#5  0x0000ffffecb8e478 in ?? () from /lib/aarch64-linux-gnu/libprotobuf.so.23
#6  0x0000fffff7fc7624 in call_init (env=0xfffffffff338, argv=0xfffffffff328, argc=1, l=<optimized out>) at ./elf/dl-init.c:70
#7  call_init (l=<optimized out>, argc=1, argv=0xfffffffff328, env=0xfffffffff338) at ./elf/dl-init.c:26
#8  0x0000fffff7fc772c in _dl_init (main_map=0xfffff7fff370, argc=1, argv=0xfffffffff328, env=0xfffffffff338) at ./elf/dl-init.c:117
#9  0x0000fffff7fd9cc8 in _dl_start_user () from /lib/ld-linux-aarch64.so.1
Backtrace stopped: previous frame identical to this frame (corrupt stack?)
```

#### 原因分析

问题发生在 /lib/aarch64-linux-gnu/libprotobuf.so.23，于是查看 nndeploy_demo_dag 的动态链接和符号表

```bash
raymond@ascenduserdg01:~/workspace/nndeploy/build$ ldd nndeploy_demo_dag | grep proto
	libprotobuf.so.23 => /lib/aarch64-linux-gnu/libprotobuf.so.23 (0x0000ffffa5990000)
raymond@ascenduserdg01:~/workspace/nndeploy/build$ nm nndeploy_demo_dag | grep protobuf | grep ' T ' | head -n 5
0000000000135bb0 T _ZN2pb11CppFeaturesC1EPN6google8protobuf5ArenaE
0000000000135bd0 T _ZN2pb11CppFeaturesC1EPN6google8protobuf5ArenaERKS0_
0000000000135bb0 T _ZN2pb11CppFeaturesC2EPN6google8protobuf5ArenaE
0000000000135bd0 T _ZN2pb11CppFeaturesC2EPN6google8protobuf5ArenaERKS0_
000000000003ea60 T _ZN6google8protobuf10DescriptorC1Ev
```

发现 demo 不仅链接了 protobuf 的动态库，因为已经存在 protobuf 中的一些符号定义，应该也链接了protobuf 的静态库；但之前 build 的结果运行并没有任何问题，发生 segment fault 是由于 build 中加入了第三方的 tokenizer，而 tokenizer 中链接了 sentencepiece 的静态库

```cmake
target_link_libraries(tokenizers_cpp PRIVATE tokenizers_c sentencepiece-static ${TOKENIZERS_CPP_LINK_LIBS})
```

sentencepiece 编译时又使用了 protobuf-lite，所以第三方的 tokenizer 库中包含了 protobuf-lite，这就导致 protobuf-lite 与 protobuf 产生了冲突

#### 临时解决方案

* 第一种方式：系统安装需要 libprotobuf，直接跳过 third_party 中 protobuf 的编译，demo 的可执行文件只链接 protobuf 的动态库
* 只修改 protobuf.cmake 如下：

```cmake
include(ExternalProject)
include(cmake/util.cmake)

if (ENABLE_NNDEPLOY_PROTOBUF STREQUAL "OFF")
elseif (ENABLE_NNDEPLOY_PROTOBUF STREQUAL "ON")
  option(protobuf_BUILD_TESTS "" OFF)
  option(protobuf_MSVC_STATIC_RUNTIME "" ${ONNX_USE_MSVC_STATIC_RUNTIME})

  find_package(PkgConfig REQUIRED)
  pkg_check_modules(PROTOBUF REQUIRED protobuf)

  find_package(Protobuf REQUIRED)
  if(Protobuf_FOUND)
    include_directories(${Protobuf_INCLUDE_DIRS})
    set(NNDEPLOY_THIRD_PARTY_LIBRARY ${NNDEPLOY_THIRD_PARTY_LIBRARY} ${Protobuf_LIBRARIES})
  else()
    set(LIBS libprotobuf)
    add_subdirectory_if_no_target(${PROJECT_SOURCE_DIR}/third_party/protobuf ${LIBS})
    include_directories(${PROBUF_ROOT}/src)
    set(NNDEPLOY_THIRD_PARTY_LIBRARY ${NNDEPLOY_THIRD_PARTY_LIBRARY} ${LIBS})
  endif()
else()
endif()
```

* 第二种方式：系统安装 libprotobuf-lite，在编译 tokenizer 的时候链接 libprotobuf-lite.so，demo 的可执行文件既链接 protobuf 的动态库同时也链接 protobuf-lite 的动态库
* 只修改 tokenizer_cpp.cmake 如下：

```cmake
include(ExternalProject)

if (ENABLE_NNDEPLOY_PLUGIN_TOKENIZER_CPP STREQUAL "OFF")
elseif (ENABLE_NNDEPLOY_PLUGIN_TOKENIZER_CPP STREQUAL "ON")
  set(TOKENZIER_CPP_PATH third_party/tokenizers-cpp)
  add_subdirectory(${TOKENZIER_CPP_PATH} tokenizers)
  include_directories(${TOKENZIER_CPP_PATH}/include)
  set(NNDEPLOY_THIRD_PARTY_LIBRARY ${NNDEPLOY_THIRD_PARTY_LIBRARY} tokenizers_cpp)   

  find_library(PROTOBUF_LITE_LIB protobuf-lite REQUIRED)
  if (PROTOBUF_LITE_LIB)
      message(STATUS "Found protobuf-lite: ${PROTOBUF_LITE_LIB}")
      set(NNDEPLOY_THIRD_PARTY_LIBRARY ${NNDEPLOY_THIRD_PARTY_LIBRARY} ${PROTOBUF_LITE_LIB})   
  else()
      message(FATAL_ERROR "protobuf-lite not found! Please install")
  endif()

else()
endif()
```

#### 说明

* 这两种解决方案都是 work around 的方式，都需要在系统上安装相关的 protobuf

* 对于第一种方式而言，第三方的 protobuf 在编译时被直接跳过，所以第三方的 protobuf 显得没有什么意义了

* 对于第二种方式来说存在风险，动态库和静态库都被链接的情况下会优先选择动态库，但依然不确定是否会发生冲突，尤其是在动态库和静态库版本不一致的情况下

## 代理失效，github连接问题等

### github 连接不稳定问题

当我们在Ascend服务器上搭建开发环境的时候会碰到仓库从github.com上clone或者其他行为卡顿特别严重的问题。对于这个现象当前有两个方法处理：

1. 使用ssh连接的方式clone仓库。

2. 通过将服务器挂上代理的方式，让https上的服务更加方便。

当前我暂且只能通过第一种方式走通服务, 第二种方式貌似遇到了一些限制等待有缘人解决了。

**对于第一种方式**主要可以参考github的这篇文章讲述[connecting-to-github-with-ssh](https://docs.github.com/en/authentication/connecting-to-github-with-ssh)。然后，走通了流程之后将会发现仍旧无法使用ssh的方式连接github，`ssh -T git@github.com`无法获得应得结果; 鉴于这个原因，题主发现默认的20端口被限制了，需要使用443端口。需要在根目录中`$HOME/.ssh/config`中增加代码，然后方可走通。

```bash
Host github.com
    Hostname ssh.github.com
    Port 443
    User git
```

除此之外，还需要在`$HOME/.gitconfig`中添加以下代码，以让后面的子模块按照ssh执行

```bash
[url "ssh://git@github.com/"]
	insteadOf = https://github.com/
```
单独使用ssh方案可以解决卡顿问题。

**对于第二种方式**可以通过两种方式在自己的服务器上布置代理：

1. 按照[该文章方式](https://www.noseeflower.icu/)从下载指定的包（对于Accend来说下载这个[arm版本的clash](https://github.com/doreamon-design/clash/releases/#:~:text=clash_2.0.24_linux_arm64.tar.gz)即可。按照文章中讲的方式去进行即可，但是可能会遇到MMDB文件缺失问题，需要自行解决，我解决的方式是把第二种方式中的Country.MMDB文件拷贝过来（MMDB文件中包含了常见的网址到ip的键值对）。
2. 下载[clash-for-linux](https://github.com/wnlen/clash-for-linux)仓库，并根据README开启代理；
   注意：这两种都是题主测试时候使用的方式，是用的linux的clash方法，需要自行购买订阅。到目前为止，题主走通了流程，但是貌似遇到了一些约束（该方法在多个服务器上已验证使用过）：![alt text](../../image/debug_record//proxy_fail.png)

题主怀疑是否是服务器只有443的端口和一些必要端口可以使用，其他端口则不行，因为如图proxy使用的52984的端口，而第一种方法中，ssh默认使用的20端口。
另题主已经尝试了网络上大部份方法得到一上结果，包括不限于手动编译git等等...


### 进入系统默认shell为sh的解决方法：

**临时方法：** 进入服务器默认进入bash：
`sudo vim /etc/passwd`
修改自己用户民末尾地方为`/usr/bin/bash`



- 编译ONNX时，absl库版本需要与onnx版本一致，否则会出现链接错误，解决办法如下：
  - 最简单的解决方案是更新 Abseil 库到最新版本，因为这可能是一个已知并修复的问题
    ```
    git clone https://github.com/abseil/abseil-cpp.git
    cd abseil-cpp
    git checkout latest
    cmake -DCMAKE_POSITION_INDEPENDENT_CODE=ON -DCMAKE_INSTALL_PREFIX=/usr/local .
    make
    sudo make install
    ```

- 本地无protobuf或者protobuf版本不匹配，编译安装nndeploy/third_party/protobuf

  下面是在 Linux 系统上编译安装最新版本 Protobuf 的详细步骤：

  1. 准备环境

      首先，确保系统已安装必要的开发工具：

      ```bash
      sudo apt update
      sudo apt install -y build-essential autoconf automake libtool curl make g++ unzip git cmake
      ```

  2. 移除现有的 Protobuf 安装

      为避免冲突，先移除系统中已有的 Protobuf：

      ```bash
      # 移除通过包管理器安装的版本
      sudo apt remove -y protobuf-compiler libprotobuf-dev

      # 如果之前手动安装过，可能需要删除这些文件
      sudo rm -rf /usr/local/include/google/protobuf
      sudo rm -rf /usr/local/bin/protoc*
      sudo rm -rf /usr/local/lib/libprotobuf*
      sudo rm -rf /usr/local/lib/libprotoc*
      sudo rm -rf /usr/local/lib/pkgconfig/protobuf*
      ```

  3. 使用 CMake 构建和安装

      推荐使用 CMake 构建，这是 Protobuf 现在推荐的方式：

      ```bash
      # 进入protobuf目录
      cd path/nndeploy/third_party/protobuf

      # 创建并进入构建目录
      mkdir build && cd build

      # 配置 CMake
      cmake .. \
          -DCMAKE_BUILD_TYPE=Release \
          -DCMAKE_POSITION_INDEPENDENT_CODE=ON \
          -DCMAKE_INSTALL_PREFIX=/usr/local \
          -Dprotobuf_BUILD_TESTS=OFF \
          -Dprotobuf_BUILD_SHARED_LIBS=ON

      # 编译
      make -j$(nproc)

      # 安装
      sudo make install

      # 更新动态链接库缓存
      sudo ldconfig
      ```

  4. 验证安装

      安装完成后，验证 Protobuf 是否正确安装：

      ```bash
      # 检查 protoc 版本
      protoc --version

      # 检查库文件是否存在
      ls -la /usr/local/lib/libprotobuf*

      # 检查头文件是否存在
      ls -la /usr/local/include/google/protobuf/
      ```

  5. 配置 pkg-config

      确保 pkg-config 能正确找到 Protobuf：

      ```bash
      # 检查 pkg-config 是否能找到 protobuf
      pkg-config --cflags --libs protobuf

      # 如果找不到，可能需要设置 PKG_CONFIG_PATH
      export PKG_CONFIG_PATH=/usr/local/lib/pkgconfig:$PKG_CONFIG_PATH
      ```

  6. 可能的问题和解决方案
  
      如果遇到链接错误，可能需要确保链接器能找到库：  

      ```bash
      # 将 /usr/local/lib 添加到链接器配置
      echo "/usr/local/lib" | sudo tee /etc/ld.so.conf.d/protobuf.conf
      sudo ldconfig
      ```
  
  7. 重新编译nndeploy

      清空build目录，重新编译nndeploy 

      ```bash
      cd path/nndeploy/build
      make clean
      cmake ..
      make -j
      make install
      ```

# 你的分析很准确！这确实是一个**动态库加载和符号管理**的问题。让我详细分析一下原因：

## 问题分析

### 1. **多DLL的NodeFactory单例问题**

从日志可以看出：
- **注册阶段**：大量节点成功注册（来自不同的DLL）
- **查找阶段**：只能找到少数几个节点类型

```cpp
// 注册时：大量节点被注册
I/nndeploy_default_str: nndeploy::dag::NodeFactory::registerNode [File xxx] register node success: nndeploy::preprocess::BatchPreprocess
// ... 很多其他节点

// 查找时：只找到很少的节点
I/nndeploy_default_str: nndeploy::dag::NodeFactory::getCreator [File xxx] node key: nndeploy::dag::Graph
I/nndeploy_default_str: nndeploy::dag::NodeFactory::getCreator [File xxx] node key: nndeploy::dag::RunningCondition
```
Search files...
### 2. **Windows DLL的符号隔离问题**

Windows DLL有严格的符号边界，不同DLL中的静态变量（包括单例）是**相互独立**的。这是问题的核心原因：

```cpp
class NodeFactory {
 public:
  static NodeFactory *getInstance() {
    static NodeFactory instance;  // 每个DLL都有自己的instance！
    return &instance;
  }
  // ...
 private:
  std::map<std::string, std::shared_ptr<NodeCreator>> creators_;
};
```

### 3. **问题详细说明**

1. **多个NodeFactory实例**：
   - `nndeploy_framework.dll` 有一个 NodeFactory 实例
   - `nndeploy_plugin.dll` 有另一个 NodeFactory 实例
   - Python扩展模块有第三个 NodeFactory 实例

2. **注册和查找分离**：
   - 各种节点在**模块加载时**注册到了各自DLL的NodeFactory实例
   - Python调用时使用的是**Python扩展模块**的NodeFactory实例
   - 导致查找时找不到其他DLL注册的节点

3. **日志证据**：
   ```bash
   # 注册时：显示所有节点都注册成功
   register node success: nndeploy::preprocess::BatchPreprocess
   register node success: nndeploy::codec::OpenCvImageDecode
   # ... 等等
   
   # 查找时：只能看到很少的节点
   node key: nndeploy::dag::Graph
   node key: nndeploy::dag::RunningCondition
   ```

## 解决方案

### 方案1：**显式导出NodeFactory符号**
在主DLL中导出NodeFactory，确保所有模块使用同一个实例：

```cpp
// 在主DLL的头文件中
NNDEPLOY_CC_API NodeFactory* GetGlobalNodeFactory();

// 在主DLL的实现中  
NNDEPLOY_CC_API NodeFactory* GetGlobalNodeFactory() {
    return NodeFactory::getInstance();
}
```

### 方案2：**使用共享内存映射**
将NodeFactory的creators_存储在共享内存中。

### 方案3：**在Python模块初始化时重新注册**
在Python模块加载时，从各个DLL中获取节点信息并重新注册：

```python
def ensure_all_nodes_registered():
    # 强制加载所有包含节点的DLL
    import ctypes
    ctypes.CDLL("./nndeploy_plugin.dll")  # 强制加载
    # 或者调用各模块的初始化函数
```

### 方案4：**修改编译配置**
使用 `/WHOLEARCHIVE` 链接选项，确保所有静态初始化代码都被执行。

### 方案5：**集中式节点注册**
创建一个专门的节点注册中心，所有节点在运行时显式注册。

**推荐使用方案1和方案3的组合**，这样可以确保所有节点都注册到同一个NodeFactory实例中。