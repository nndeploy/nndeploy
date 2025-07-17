# Installation Guide

1. Install OpenCV (Skip if already installed)
   Execute in SDK directory:

   ```bash
   python3 install_opencv.py
   ```

   This script will automatically download and install OpenCV 4.8.0.

2. Set Environment Variables and Paths
   Execute in SDK directory:

   ```bash
   source ./set_install_env.sh  # Set third-party library paths
   ```

3. Run Demo Programs
   ```bash
   ./demo/nndeploy_demo_dag  # Run DAG demo program
   ```

Notes:
- Make sure CMake, Python3 and other basic development tools are installed
- For conda users, you may need to fix system library conflicts using fixed_sys_conda.sh

# 安装说明

1. 安装OpenCV (如果已安装可跳过此步骤)
   在SDK目录下执行:

   ```bash
   python3 install_opencv.py
   ```

   该脚本会自动下载并编译安装OpenCV 4.8.0版本。

2. 设置环境变量和路径
   在SDK目录下执行:

   ```bash
   source ./set_install_env.sh  # 设置第三方库路径
   ```

3. 运行示例程序
   ```bash
   ./demo/nndeploy_demo_dag  # 运行DAG示例程序
   ```

注意事项:
- 确保系统已安装CMake、Python3等基础开发工具
- 对于conda用户，可能需要使用fixed_sys_conda.sh修复系统库冲突
