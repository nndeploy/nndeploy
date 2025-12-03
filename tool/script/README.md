# Installation Guide

1. Install OpenCV (Skip if already installed)
   Execute in SDK directory:

   ```bash
   python3 install_opencv.py
   ```

   This script will automatically download and install OpenCV 4.10.0.

2. Run Demo Programs
   Execute in SDK directory:

   **Linux:**
   ```bash
   source ./tool/script/set_install_env.sh  # Set third-party library paths
   ./demo/nndeploy_demo_dag  # Run DAG demo program
   ```

   **Windows:**
   ```cmd
   .\tool\script\set_install_env.bat  # Set third-party library paths
   ```

Notes:
- Make sure CMake, Python3 and other basic development tools are installed
- For conda users, you may need to fix system library conflicts using fixed_sys_conda.sh
