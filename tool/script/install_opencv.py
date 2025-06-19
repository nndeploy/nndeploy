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

# 获取当前执行目录
WORKSPACE = Path(os.getcwd())
OPENCV_BUILD_DIR = WORKSPACE / "temp"
OPENCV_DIR = "opencv" + OPENCV_VER
OPENCV_INSTALL_DIR = WORKSPACE / "third_party" / OPENCV_DIR
print(OPENCV_INSTALL_DIR)

# Create third party library directory
OPENCV_BUILD_DIR.mkdir(parents=True, exist_ok=True)

# Change to third party library directory
os.chdir(OPENCV_BUILD_DIR)

# Download OpenCV source code
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
os.rename(f"opencv-{OPENCV_VER}", "opencv")

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
        -DCMAKE_INSTALL_PREFIX="{OPENCV_INSTALL_DIR}"
    """
else:
    cmake_cmd = f"""cmake .. \
        -DBUILD_TESTS=OFF \
        -DBUILD_PERF_TESTS=OFF \
        -DWITH_FFMPEG=ON \
        -DCMAKE_INSTALL_PREFIX="{OPENCV_INSTALL_DIR}"
    """
os.system(cmake_cmd)

# Build and install
print("Building OpenCV...")
os.system("cmake --build . --config Release -j6")
os.system("cmake --install . --config Release")

print(f"OpenCV {OPENCV_VER} installation completed!")
