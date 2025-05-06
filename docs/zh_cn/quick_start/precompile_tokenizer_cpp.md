在Ubuntu下安装Rust环境主要有以下步骤：

### 安装前准备
1. 更新系统软件包列表，确保系统是最新的：
```bash
sudo apt update
sudo apt upgrade
```
2. 安装必要的编译工具和依赖库：
```bash
sudo apt install curl build-essential gcc make
```

### 安装Rust
1. **使用官方脚本安装**：执行以下命令下载并运行Rust的安装脚本。
```bash
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
```
运行该命令后，安装脚本会提示你进行一些选择，通常选择默认选项1即可完成安装。
2. **使用镜像源安装**：如果官方安装速度较慢，可以使用国内镜像源来加速安装。
```bash
export RUSTUP_DIST_SERVER=https://mirrors.ustc.edu.cn/rust-static
export RUSTUP_UPDATE_ROOT=https://mirrors.ustc.edu.cn/rust-static/rustup
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
```

### 配置环境变量
安装完成后，需要将Rust的相关路径添加到环境变量中，以便系统能够找到Rust的工具。可以使用以下命令：
```bash
source $HOME/.cargo/env
```

### 验证安装
可以通过检查Rust编译器（rustc）和包管理器（cargo）的版本来验证是否安装成功：
```bash
rustc --version
cargo --version
```
若出现版本信息，则表示安装成功。

### 更近一步的编译参考官方文档
[tokenizer-cpp](https://github.com/mlc-ai/tokenizers-cpp)