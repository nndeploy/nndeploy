#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
import shutil
import zipfile
import requests
import platform
from pathlib import Path

# OpenCV version
OPENCV_VER = "4.8.0"

# 
WORKSPACE = Path(os.getcwd())
OPENCV_BUILD_DIR = WORKSPACE / "download"
OPENCV_DIR = "opencv" + OPENCV_VER
OPENCV_INSTALL_DIR = WORKSPACE / "third_party" / OPENCV_DIR
print(OPENCV_INSTALL_DIR)

# Create third party library directory
OPENCV_BUILD_DIR.mkdir(parents=True, exist_ok=True)

# Change to third party library directory
os.chdir(OPENCV_BUILD_DIR)

# # Download OpenCV source code
url = f"https://github.com/opencv/opencv/archive/refs/tags/{OPENCV_VER}.zip"
filename = url.split('/')[-1]

print(f"Downloading OpenCV {OPENCV_VER}...")
response = requests.get(url, stream=True)
with open(filename, 'wb') as f:
    for chunk in response.iter_content(chunk_size=8192):
        f.write(chunk)

# Extract source code
print("Extracting source code...")
with zipfile.ZipFile(filename, 'r') as zip_ref:
    zip_ref.extractall(".")

# Rename directory
if (OPENCV_BUILD_DIR / "opencv").exists():
    shutil.rmtree(OPENCV_BUILD_DIR / "opencv")
# rename
try:
    os.rename(f"opencv-{OPENCV_VER}", "opencv")
except PermissionError:
    shutil.move(f"opencv-{OPENCV_VER}", "opencv")

# Create and change to build directory
os.chdir("opencv")
os.makedirs("build", exist_ok=True)
os.chdir("build")

# Configure CMake based on platform
print("Configuring CMake...")
if platform.system() == "Windows":
    cmake_cmd = f"""cmake .. -A x64 -T v142 \
        -DBUILD_TESTS=OFF \
        -DBUILD_PERF_TESTS=OFF \
        -DWITH_FFMPEG=ON \
        -DBUILD_opencv_world=OFF \
        -DBUILD_opencv_core=ON \
        -DBUILD_opencv_imgproc=ON \
        -DBUILD_opencv_imgcodecs=ON \
        -DBUILD_opencv_videoio=ON \
        -DBUILD_opencv_highgui=ON \
        -DBUILD_opencv_video=ON \
        -DBUILD_opencv_dnn=ON \
        -DBUILD_opencv_calib3d=ON \
        -DBUILD_opencv_features2d=ON \
        -DBUILD_opencv_flann=ON \
        -DBUILD_opencv_apps=OFF \
        -DBUILD_opencv_java=OFF \
        -DBUILD_opencv_js=OFF \
        -DBUILD_opencv_python2=OFF \
        -DBUILD_opencv_python3=OFF \
        -DBUILD_EXAMPLES=OFF \
        -DBUILD_DOCS=OFF \
        -DWITH_PROTOBUF=OFF \
        -DBUILD_PROTOBUF=OFF \
        -DCMAKE_INSTALL_PREFIX="{OPENCV_INSTALL_DIR}"
    """
else:
    cmake_cmd = f"""cmake .. \
        -DBUILD_TESTS=OFF \
        -DBUILD_PERF_TESTS=OFF \
        -DWITH_FFMPEG=ON \
        -DBUILD_opencv_world=OFF \
        -DBUILD_opencv_core=ON \
        -DBUILD_opencv_imgproc=ON \
        -DBUILD_opencv_imgcodecs=ON \
        -DBUILD_opencv_videoio=ON \
        -DBUILD_opencv_highgui=ON \
        -DBUILD_opencv_video=ON \
        -DBUILD_opencv_dnn=ON \
        -DBUILD_opencv_calib3d=ON \
        -DBUILD_opencv_features2d=ON \
        -DBUILD_opencv_flann=ON \
        -DBUILD_opencv_apps=OFF \
        -DBUILD_opencv_java=OFF \
        -DBUILD_opencv_js=OFF \
        -DBUILD_opencv_python2=OFF \
        -DBUILD_opencv_python3=OFF \
        -DBUILD_EXAMPLES=OFF \
        -DBUILD_DOCS=OFF \
        -DWITH_PROTOBUF=OFF \
        -DBUILD_PROTOBUF=OFF \
        -DCMAKE_INSTALL_PREFIX="{OPENCV_INSTALL_DIR}"
    """
os.system(cmake_cmd)

# Build and install
print("Building OpenCV...")
os.system("cmake --build . --config Release -j6")
os.system("cmake --install . --config Release")

if platform.system() == "Windows":
    # 创建目标目录结构
    target_lib_dir = OPENCV_INSTALL_DIR / "lib"
    target_bin_dir = OPENCV_INSTALL_DIR / "bin"
    
    # 确保目录存在
    target_lib_dir.mkdir(parents=True, exist_ok=True)
    target_bin_dir.mkdir(parents=True, exist_ok=True)
    
    # 拷贝lib目录内容
    opencv_lib_dir = OPENCV_INSTALL_DIR / "x64" / "vc16" / "lib"
    if opencv_lib_dir.exists():
        for lib_file in opencv_lib_dir.glob("*"):
            if lib_file.is_file():
                shutil.copy2(lib_file, target_lib_dir)
    
    # 拷贝bin目录内容
    opencv_bin_dir = OPENCV_INSTALL_DIR / "x64" / "vc16" / "bin"
    if opencv_bin_dir.exists():
        for bin_file in opencv_bin_dir.glob("*"):
            if bin_file.is_file():
                shutil.copy2(bin_file, target_bin_dir)

print(f"OpenCV {OPENCV_VER} installation completed!")
