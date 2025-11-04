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
        -DBUILD_ZLIB=OFF \
        -DBUILD_PNG=ON \
        -DBUILD_JPEG=ON \
        -DBUILD_TIFF=ON \
        -DBUILD_WEBP=OFF \
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
elif platform.system() == "Darwin":
    cmake_cmd = f"""cmake .. \
        -DBUILD_TESTS=OFF \
        -DBUILD_PERF_TESTS=OFF \
        -DWITH_FFMPEG=ON \
        -DBUILD_ZLIB=OFF \
        -DBUILD_PNG=ON \
        -DBUILD_JPEG=ON \
        -DBUILD_TIFF=ON \
        -DBUILD_WEBP=OFF \
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
        -DBUILD_ZLIB=OFF \
        -DBUILD_PNG=ON \
        -DBUILD_JPEG=ON \
        -DBUILD_TIFF=ON \
        -DBUILD_WEBP=OFF \
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
    # Create target directory structure
    target_lib_dir = OPENCV_INSTALL_DIR / "lib"
    target_bin_dir = OPENCV_INSTALL_DIR / "bin"
    
    # Ensure directories exist
    target_lib_dir.mkdir(parents=True, exist_ok=True)
    target_bin_dir.mkdir(parents=True, exist_ok=True)
    
    # Copy lib directory contents
    opencv_lib_dir = OPENCV_INSTALL_DIR / "x64" / "vc16" / "lib"
    if opencv_lib_dir.exists():
        for lib_file in opencv_lib_dir.glob("*"):
            if lib_file.is_file():
                shutil.copy2(lib_file, target_lib_dir, follow_symlinks=False)
    
    # Copy bin directory contents
    opencv_bin_dir = OPENCV_INSTALL_DIR / "x64" / "vc16" / "bin"
    if opencv_bin_dir.exists():
        for bin_file in opencv_bin_dir.glob("*"):
            if bin_file.is_file():
                shutil.copy2(bin_file, target_bin_dir, follow_symlinks=False)
                
# Check and fix lib64 directory issue
lib64_dir = OPENCV_INSTALL_DIR / "lib64"
lib_dir = OPENCV_INSTALL_DIR / "lib"

if lib64_dir.exists() and not lib_dir.exists():
    print("Detected lib64 directory, renaming it to lib directory...")
    shutil.move(str(lib64_dir), str(lib_dir))
    print("Successfully renamed lib64 directory to lib directory")
elif lib64_dir.exists() and lib_dir.exists():
    print("Detected both lib64 and lib directories, merging lib64 contents into lib directory...")
    for lib_file in lib64_dir.iterdir():
        if lib_file.is_file():
            shutil.copy2(lib_file, lib_dir, follow_symlinks=False)
        elif lib_file.is_dir():
            target_dir = lib_dir / lib_file.name
            if target_dir.exists():
                shutil.rmtree(target_dir)
            shutil.copytree(lib_file, target_dir, symlinks=True)
    shutil.rmtree(lib64_dir)
    print("Successfully merged lib64 directory contents into lib directory")

# Verify installation
print("Verifying installation results:")
include_dir = OPENCV_INSTALL_DIR / "include"
lib_dir = OPENCV_INSTALL_DIR / "lib"

print(f"Header files directory: {include_dir} - {'exists' if include_dir.exists() else 'not found'}")
print(f"Library files directory: {lib_dir} - {'exists' if lib_dir.exists() else 'not found'}")

if include_dir.exists():
    print("Header files:")
    for header in include_dir.glob("**/*.h*"):
        print(f"  {header.relative_to(include_dir)}")

if lib_dir.exists():
    print("Library files:")
    for lib_file in lib_dir.iterdir():
        if lib_file.is_file():
            print(f"  {lib_file.name}")
            
print(f"OpenCV {OPENCV_VER} installation completed!")
