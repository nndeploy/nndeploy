#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
import shutil
import zipfile
import tarfile
import requests
import platform
from pathlib import Path

# ONNX Runtime version
ONNXRUNTIME_VER = "1.18.0"

# 获取当前执行目录
WORKSPACE = Path(os.getcwd())
ONNXRUNTIME_BUILD_DIR = WORKSPACE / "download"
ONNXRUNTIME_DIR = "onnxruntime" + ONNXRUNTIME_VER
ONNXRUNTIME_INSTALL_DIR = WORKSPACE / "third_party" / ONNXRUNTIME_DIR
print(ONNXRUNTIME_INSTALL_DIR)

# Create third party library directory
ONNXRUNTIME_BUILD_DIR.mkdir(parents=True, exist_ok=True)

# Change to third party library directory
os.chdir(ONNXRUNTIME_BUILD_DIR)

# 根据平台确定下载URL和文件名
def get_download_info():
    """根据当前平台返回对应的下载信息"""
    system = platform.system()
    machine = platform.machine()
    
    if system == "Windows":
        if machine == "AMD64" or machine == "x86_64":
            filename = f"onnxruntime-win-x64-{ONNXRUNTIME_VER}.zip"
            url = f"https://github.com/microsoft/onnxruntime/releases/download/v{ONNXRUNTIME_VER}/{filename}"
        else:
            raise ValueError(f"不支持的Windows架构: {machine}")
    elif system == "Linux":
        if machine == "x86_64":
            filename = f"onnxruntime-linux-x64-{ONNXRUNTIME_VER}.tgz"
            url = f"https://github.com/microsoft/onnxruntime/releases/download/v{ONNXRUNTIME_VER}/{filename}"
        elif machine in ["aarch64", "arm64"]:
            filename = f"onnxruntime-linux-aarch64-{ONNXRUNTIME_VER}.tgz"
            url = f"https://github.com/microsoft/onnxruntime/releases/download/v{ONNXRUNTIME_VER}/{filename}"
        else:
            raise ValueError(f"不支持的Linux架构: {machine}")
    elif system == "Darwin":  # macOS
        if machine == "x86_64":
            filename = f"onnxruntime-osx-x86_64-{ONNXRUNTIME_VER}.tgz"
            url = f"https://github.com/microsoft/onnxruntime/releases/download/v{ONNXRUNTIME_VER}/{filename}"
        elif machine in ["arm64", "aarch64"]:
            filename = f"onnxruntime-osx-arm64-{ONNXRUNTIME_VER}.tgz"
            url = f"https://github.com/microsoft/onnxruntime/releases/download/v{ONNXRUNTIME_VER}/{filename}"
        else:
            raise ValueError(f"不支持的macOS架构: {machine}")
    else:
        raise ValueError(f"不支持的操作系统: {system}")
    
    return url, filename

# 获取下载信息
url, filename = get_download_info()

# 下载ONNX Runtime预编译版本
print(f"正在下载ONNX Runtime {ONNXRUNTIME_VER}...")
print(f"下载URL: {url}")
response = requests.get(url, stream=True)
if response.status_code != 200:
    raise Exception(f"下载失败，状态码: {response.status_code}")

with open(filename, 'wb') as f:
    for chunk in response.iter_content(chunk_size=8192):
        f.write(chunk)

# 解压文件
print("正在解压文件...")
if filename.endswith('.zip'):
    with zipfile.ZipFile(filename, 'r') as zip_ref:
        zip_ref.extractall(".")
elif filename.endswith('.tgz') or filename.endswith('.tar.gz'):
    with tarfile.open(filename, 'r:gz') as tar_ref:
        tar_ref.extractall(".")
else:
    raise ValueError(f"不支持的文件格式: {filename}")

# 查找解压后的目录
extracted_dirs = [d for d in os.listdir('.') if os.path.isdir(d) and 'onnxruntime' in d and d != 'onnxruntime']
if not extracted_dirs:
    raise Exception("未找到解压后的ONNX Runtime目录")

extracted_dir = extracted_dirs[0]
print(f"找到解压目录: {extracted_dir}")

# 重命名目录
if os.path.exists("onnxruntime"):
    shutil.rmtree("onnxruntime")

try:
    os.rename(extracted_dir, "onnxruntime")
except PermissionError:
    shutil.move(extracted_dir, "onnxruntime")

# 安装到目标目录
print(f"正在安装到目标目录: {ONNXRUNTIME_INSTALL_DIR}")
if ONNXRUNTIME_INSTALL_DIR.exists():
    shutil.rmtree(ONNXRUNTIME_INSTALL_DIR)

# 拷贝整个目录
shutil.copytree("onnxruntime", ONNXRUNTIME_INSTALL_DIR)

# 验证安装
include_dir = ONNXRUNTIME_INSTALL_DIR / "include"
lib_dir = ONNXRUNTIME_INSTALL_DIR / "lib"

if platform.system() == "Windows":
    lib_dir = ONNXRUNTIME_INSTALL_DIR / "lib"
    if not lib_dir.exists():
        # Windows版本可能在根目录下有lib文件
        for item in ONNXRUNTIME_INSTALL_DIR.iterdir():
            if item.suffix in ['.lib', '.dll']:
                lib_dir.mkdir(exist_ok=True)
                break

print("验证安装结果:")
print(f"头文件目录: {include_dir} - {'存在' if include_dir.exists() else '不存在'}")
print(f"库文件目录: {lib_dir} - {'存在' if lib_dir.exists() else '不存在'}")

if include_dir.exists():
    print("头文件列表:")
    for header in include_dir.glob("**/*.h"):
        print(f"  {header.relative_to(include_dir)}")

if lib_dir.exists():
    print("库文件列表:")
    for lib_file in lib_dir.iterdir():
        if lib_file.is_file():
            print(f"  {lib_file.name}")

print(f"ONNX Runtime {ONNXRUNTIME_VER} 安装完成!")
