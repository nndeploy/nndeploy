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



# 设置标准输出编码为UTF-8，解决Windows下中文输出问题
if platform.system() == "Windows":
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

# MNN version
# MNN_VER = "2.4.0"
MNN_VER = "3.2.4"

# 获取当前执行目录
WORKSPACE = Path(os.getcwd())
MNN_BUILD_DIR = WORKSPACE / "download"
MNN_DIR = "mnn" + MNN_VER
MNN_INSTALL_DIR = WORKSPACE / "third_party" / MNN_DIR
print(MNN_INSTALL_DIR)

# Create third party library directory
MNN_BUILD_DIR.mkdir(parents=True, exist_ok=True)

# Change to third party library directory
os.chdir(MNN_BUILD_DIR)

# 根据平台确定下载URL和文件名
def get_download_info(system, machine):
    """根据当前平台返回对应的下载信息"""
    # system = platform.system()
    # machine = platform.machine()
    
    if system == "Windows":
        if machine == "AMD64" or machine == "x86_64":
            filename = f"mnn_{MNN_VER}_windows_x64_cpu_opencl.zip"
            url = f"https://github.com/alibaba/MNN/releases/download/{MNN_VER}/{filename}"
        else:
            raise ValueError(f"Unsupported Windows architecture: {machine}")
    elif system == "Linux":
        if machine == "x86_64":
            filename = f"mnn_{MNN_VER}_linux_x64_cpu_opencl.zip"
            url = f"https://github.com/alibaba/MNN/releases/download/{MNN_VER}/{filename}"
        else:
            raise ValueError(f"Unsupported Linux architecture: {machine}")
    elif system == "Darwin":  # macOS
        if machine in ["arm64", "aarch64"]:
            filename = f"mnn_{MNN_VER}_macos_x64_arm82_cpu_opencl_metal.zip"
            url = f"https://github.com/alibaba/MNN/releases/download/{MNN_VER}/{filename}"
        elif machine == "x86_64":
            filename = f"mnn_{MNN_VER}_macos_x64_arm82_cpu_opencl_metal.zip"
            url = f"https://github.com/alibaba/MNN/releases/download/{MNN_VER}/{filename}"
        else:
            raise ValueError(f"Unsupported macOS architecture: {machine}")
    elif system == "Android":
        filename = f"mnn_{MNN_VER}_android_armv7_armv8_cpu_opencl_vulkan.zip"
        url = f"https://github.com/alibaba/MNN/releases/download/{MNN_VER}/{filename}"
    elif system == "iOS":
        if machine in ["arm64", "aarch64"]:
            filename = f"mnn_{MNN_VER}_ios_armv82_cpu_metal_coreml.zip"
            url = f"https://github.com/alibaba/MNN/releases/download/{MNN_VER}/{filename}"
        else:
            raise ValueError(f"Unsupported iOS architecture: {machine}")
    else:
        raise ValueError(f"Unsupported operating system: {system}")
    
    return url, filename

# 获取下载信息
# 允许外部指定系统和架构信息，如果未指定则使用当前平台信息
import argparse

# 解析命令行参数
parser = argparse.ArgumentParser(description='Install MNN precompiled library')
parser.add_argument('--system', type=str, help='Target system (Windows, Linux, Darwin, Android, iOS)')
parser.add_argument('--machine', type=str, help='Target machine architecture (x86_64, arm64, etc.)')
args = parser.parse_args()

# 如果命令行指定了系统和架构，使用指定的值；否则使用当前平台信息
if args.system:
    system = args.system
else:
    system = platform.system()

if args.machine:
    machine = args.machine
else:
    machine = platform.machine()

url, filename = get_download_info(system, machine)

# 下载MNN预编译版本
print(f"Downloading MNN {MNN_VER}...")
print(f"Download URL: {url}")
response = requests.get(url, stream=True)
if response.status_code != 200:
    raise Exception(f"Download failed, status code: {response.status_code}")

with open(filename, 'wb') as f:
    for chunk in response.iter_content(chunk_size=8192):
        f.write(chunk)

# 解压文件
print("Extracting files...")
if filename.endswith('.zip'):
    with zipfile.ZipFile(filename, 'r') as zip_ref:
        zip_ref.extractall(".")
elif filename.endswith('.tgz') or filename.endswith('.tar.gz'):
    with tarfile.open(filename, 'r:gz') as tar_ref:
        tar_ref.extractall(".")
else:
    raise ValueError(f"Unsupported file format: {filename}")

# 查找解压后的目录
extracted_dirs = [d for d in os.listdir('.') if os.path.isdir(d) and 'mnn' in d.lower() and d != 'mnn']
if not extracted_dirs:
    raise Exception("Extracted MNN directory not found")

extracted_dir = extracted_dirs[0]
print(f"Found extracted directory: {extracted_dir}")

# 重命名目录
if os.path.exists("mnn"):
    shutil.rmtree("mnn")

try:
    os.rename(extracted_dir, "mnn")
except PermissionError:
    shutil.move(extracted_dir, "mnn")

# 下载源代码以获取头文件
print(f"Downloading MNN source code for headers...")
source_url = f"https://github.com/alibaba/MNN/archive/refs/tags/{MNN_VER}.tar.gz"
source_filename = f"MNN-{MNN_VER}.tar.gz"

print(f"Downloading source from: {source_url}")
response = requests.get(source_url, stream=True)
if response.status_code != 200:
    raise Exception(f"Source download failed, status code: {response.status_code}")

with open(source_filename, 'wb') as f:
    for chunk in response.iter_content(chunk_size=8192):
        f.write(chunk)

# 解压源代码
print("Extracting source code...")
with tarfile.open(source_filename, 'r:gz') as tar_ref:
    tar_ref.extractall(".")

# 查找源代码目录
source_dirs = [d for d in os.listdir('.') if os.path.isdir(d) and d.startswith(f'MNN-{MNN_VER}')]
if not source_dirs:
    raise Exception("Source directory not found")

source_dir = source_dirs[0]
print(f"Found source directory: {source_dir}")

# 创建include目录并拷贝头文件
mnn_include_dir = Path("mnn") / "include"
mnn_include_dir.mkdir(parents=True, exist_ok=True)

# 从源代码拷贝include目录
source_include_dir = Path(source_dir) / "include"
if source_include_dir.exists():
    print("Copying header files from source...")
    shutil.copytree(source_include_dir, mnn_include_dir, symlinks=True, dirs_exist_ok=True)
else:
    print("Warning: include directory not found in source code")

# 安装到目标目录
print(f"Installing to target directory: {MNN_INSTALL_DIR}")
if MNN_INSTALL_DIR.exists():
    shutil.rmtree(MNN_INSTALL_DIR)

# 拷贝整个目录
shutil.copytree("mnn", MNN_INSTALL_DIR, symlinks=True)

# 验证安装
all_libs = MNN_INSTALL_DIR.rglob("*")
print(f"all_libs: {all_libs}")

include_dir = MNN_INSTALL_DIR / "include"
lib_dir = MNN_INSTALL_DIR / "lib"  
bin_dir = MNN_INSTALL_DIR / "bin"

# 从all_libs中移除include_dir、lib_dir和bin_dir
all_libs = [lib for lib in all_libs if lib not in [include_dir, lib_dir, bin_dir]]
print(f"all_libs: {all_libs}")

# 创建lib和bin目录
lib_dir.mkdir(parents=True, exist_ok=True)
bin_dir.mkdir(parents=True, exist_ok=True)

# 拷贝动态库文件到相应目录
print("Copying dynamic libraries...")
for item in all_libs:
    print(f"item: {item}")
    if item.is_file():
        if system == "Windows":
            # Windows平台：.dll文件拷贝到bin目录，.lib文件拷贝到lib目录
            if item.suffix == '.dll':
                shutil.copy2(item, bin_dir, follow_symlinks=False)
                print(f"Copied {item.name} to bin directory")
            elif item.suffix == '.lib':
                shutil.copy2(item, lib_dir, follow_symlinks=False)
                print(f"Copied {item.name} to lib directory")
        elif system == "Android":
            if item.suffix in ['.so', '.dylib'] or '.so.' in item.name:
               # 检查文件路径中是否包含架构信息，拷贝到对应的架构目录
               if "armeabi-v7a" in str(item):
                   target_lib_dir = MNN_INSTALL_DIR / "lib" / "armeabi-v7a"
                   target_lib_dir.mkdir(parents=True, exist_ok=True)
                   shutil.copy2(item, target_lib_dir, follow_symlinks=False)
                   print(f"Copied {item.name} to armeabi-v7a lib directory")
               elif "arm64-v8a" in str(item):
                   target_lib_dir = MNN_INSTALL_DIR / "lib" / "arm64-v8a"
                   target_lib_dir.mkdir(parents=True, exist_ok=True)
                   shutil.copy2(item, target_lib_dir, follow_symlinks=False)
                   print(f"Copied {item.name} to arm64-v8a lib directory")
               else:
                   # 如果没有架构信息，拷贝到默认的lib目录
                   shutil.copy2(item, lib_dir, follow_symlinks=False)
                   print(f"Copied {item.name} to lib directory")
        else:
            # Linux/macOS平台：.so/.dylib文件拷贝到lib目录
            if item.suffix in ['.so', '.dylib'] or '.so.' in item.name:
                shutil.copy2(item, lib_dir, follow_symlinks=False)
                print(f"Copied {item.name} to lib directory")

print("Verifying installation:")
print(f"Include directory: {include_dir} - {'exists' if include_dir.exists() else 'not found'}")
print(f"Library directory: {lib_dir} - {'exists' if lib_dir.exists() else 'not found'}")
print(f"Binary directory: {bin_dir} - {'exists' if bin_dir.exists() else 'not found'}")

if include_dir.exists():
    print("Header files:")
    for header in include_dir.rglob("*"):
        if header.is_file():
            print(f"  {header.relative_to(include_dir)}")

if lib_dir.exists():
    print("Library files:")
    for lib_file in lib_dir.rglob("*"):
        if lib_file.is_file():
            print(f"  {lib_file.relative_to(lib_dir)}")

if bin_dir.exists():
    print("Binary files:")
    for bin_file in bin_dir.rglob("*"):
        if bin_file.is_file():
            print(f"  {bin_file.relative_to(bin_dir)}")

print(f"MNN {MNN_VER} installation completed!")
