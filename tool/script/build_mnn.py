#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
import shutil
import subprocess
import platform
from pathlib import Path
import argparse

def run_command(cmd, cwd=None, check=True):
    """执行命令并打印输出"""
    print(f"Running: {' '.join(cmd) if isinstance(cmd, list) else cmd}")
    if isinstance(cmd, str):
        cmd = cmd.split()
    
    result = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True)
    
    if result.stdout:
        print(result.stdout)
    if result.stderr:
        print(result.stderr, file=sys.stderr)
    
    if check and result.returncode != 0:
        raise subprocess.CalledProcessError(result.returncode, cmd)
    
    return result

# 设置标准输出编码为UTF-8，解决Windows下中文输出问题
if platform.system() == "Windows":
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

# MNN version
MNN_VER = "3.2.4"

# 获取当前执行目录
WORKSPACE = Path(os.getcwd())
MNN_BUILD_DIR = WORKSPACE / "download"
MNN_SOURCE_DIR = MNN_BUILD_DIR / f"MNN-{MNN_VER}"
MNN_INSTALL_DIR = WORKSPACE / "third_party" / f"mnn{MNN_VER}"

print(f"Workspace: {WORKSPACE}")
print(f"Build directory: {MNN_BUILD_DIR}")
print(f"Source directory: {MNN_SOURCE_DIR}")
print(f"Install directory: {MNN_INSTALL_DIR}")

# 解析命令行参数
parser = argparse.ArgumentParser(description='Build and install MNN from source')
parser.add_argument('--system', type=str, help='Target system (Windows, Linux, Darwin, Android, iOS)')
parser.add_argument('--machine', type=str, help='Target machine architecture (x86_64, arm64, etc.)')
parser.add_argument('--build-type', type=str, default='Release', choices=['Debug', 'Release'], 
                   help='Build type (Debug or Release)')
parser.add_argument('--enable-opencl', action='store_true', help='Enable OpenCL support')
parser.add_argument('--enable-vulkan', action='store_true', help='Enable Vulkan support')
parser.add_argument('--enable-metal', action='store_true', help='Enable Metal support (macOS/iOS)')
parser.add_argument('--enable-coreml', action='store_true', help='Enable CoreML support (macOS/iOS)')
parser.add_argument('--android-ndk', type=str, default='$ANDROID_NDK_ROOT', help='Android NDK path (for Android builds)')
parser.add_argument('--android-abi', type=str, default='arm64-v8a', 
                   choices=['armeabi-v7a', 'arm64-v8a', 'x86', 'x86_64'],
                   help='Android ABI (for Android builds)')
parser.add_argument('--ios-deployment-target', type=str, default='11.0',
                   help='iOS deployment target (for iOS builds)')
args = parser.parse_args()

# 如果命令行指定了系统和架构，使用指定的值；否则使用当前平台信息
system = args.system if args.system else platform.system()
machine = args.machine if args.machine else platform.machine()

print(f"Target system: {system}")
print(f"Target machine: {machine}")
print(f"Build type: {args.build_type}")

# 创建构建目录
MNN_BUILD_DIR.mkdir(parents=True, exist_ok=True)
os.chdir(MNN_BUILD_DIR)

# 下载源代码
print(f"Downloading MNN {MNN_VER} source code...")
source_url = f"https://github.com/alibaba/MNN/archive/refs/tags/{MNN_VER}.tar.gz"
source_filename = f"MNN-{MNN_VER}.tar.gz"

if not Path(source_filename).exists():
    print(f"Downloading from: {source_url}")
    import requests
    response = requests.get(source_url, stream=True)
    if response.status_code != 200:
        raise Exception(f"Source download failed, status code: {response.status_code}")
    
    with open(source_filename, 'wb') as f:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)
    print("Download completed")
else:
    print("Source archive already exists, skipping download")

# 解压源代码
if not MNN_SOURCE_DIR.exists():
    print("Extracting source code...")
    import tarfile
    with tarfile.open(source_filename, 'r:gz') as tar_ref:
        tar_ref.extractall(".")
    print("Extraction completed")
else:
    print("Source directory already exists, skipping extraction")

# 进入源代码目录
os.chdir(MNN_SOURCE_DIR)

# 创建构建目录
build_dir = MNN_SOURCE_DIR / "build"
build_dir.mkdir(exist_ok=True)
os.chdir(build_dir)

# 构建CMake命令
cmake_args = [
    "cmake",
    "..",
    f"-DCMAKE_BUILD_TYPE={args.build_type}",
    f"-DCMAKE_INSTALL_PREFIX={MNN_INSTALL_DIR}",
    "-DMNN_BUILD_SHARED_LIBS=ON",
    "-DMNN_LOW_MEMORY=true",
    "-DMNN_CPU_WEIGHT_DEQUANT_GEMM=true",
    "-DMNN_BUILD_LLM=true",
    "-DMNN_SUPPORT_TRANSFORMER_FUSE=true",
]

# 根据目标平台添加特定的CMake参数
if system == "Windows":
    cmake_args.extend([
        "-G", "Visual Studio 16 2019" if shutil.which("cmake") else "MinGW Makefiles",
        "-A", "x64" if machine in ["AMD64", "x86_64"] else "Win32",
        "-DMNN_AVX512=true"
    ])
elif system == "Android":
    if args.android_ndk == '$ANDROID_NDK_ROOT':
        args.android_ndk = os.environ.get('ANDROID_NDK_ROOT')
        if not args.android_ndk:
            raise ValueError("ANDROID_NDK_ROOT environment variable is not set")
    if not args.android_ndk:
        raise ValueError("Android NDK path is required for Android builds (use --android-ndk)")
    
    cmake_args.extend([
        f"-DCMAKE_TOOLCHAIN_FILE={args.android_ndk}/build/cmake/android.toolchain.cmake",
        f"-DANDROID_ABI={args.android_abi}",
        "-DANDROID_PLATFORM=android-21",
        "-DMNN_BUILD_FOR_ANDROID_COMMAND=ON",
        "-DANDROID_STL=c++_static",
        "-DMNN_ARM82=true",
        "-DMNN_OPENCL=true",
        "-DMNN_USE_LOGCAT=true",
        "-DMNN_USE_SSE=OFF",
        "-DNATIVE_LIBRARY_OUTPUT=. -DNATIVE_INCLUDE_OUTPUT=. $*"
    ])
elif system == "iOS":
    cmake_args.extend([
        "-G", "Xcode",
        "-DCMAKE_TOOLCHAIN_FILE=../cmake/ios.toolchain.cmake",
        "-DPLATFORM=OS64",
        f"-DDEPLOYMENT_TARGET={args.ios_deployment_target}",
        "-DENABLE_BITCODE=OFF",
    ])

# 添加后端支持
if args.enable_opencl:
    cmake_args.append("-DMNN_OPENCL=ON")
if args.enable_vulkan:
    cmake_args.append("-DMNN_VULKAN=ON")
if args.enable_metal and system in ["Darwin", "iOS"]:
    cmake_args.append("-DMNN_METAL=ON")
if args.enable_coreml and system in ["Darwin", "iOS"]:
    cmake_args.append("-DMNN_COREML=ON")

# 运行CMake配置
print("Configuring with CMake...")
run_command(cmake_args)

# 构建项目
print("Building MNN...")
build_cmd = ["cmake", "--build", ".", "--config", args.build_type]
if system != "Windows":
    # 在非Windows平台使用并行构建
    import multiprocessing
    build_cmd.extend(["-j", str(multiprocessing.cpu_count())])

run_command(build_cmd)

# 安装到目标目录
print(f"Installing to: {MNN_INSTALL_DIR}")
if MNN_INSTALL_DIR.exists():
    shutil.rmtree(MNN_INSTALL_DIR)

install_cmd = ["cmake", "--build", ".", "--target", "install", "--config", args.build_type]
run_command(install_cmd)

# 验证安装
print("Verifying installation:")
include_dir = MNN_INSTALL_DIR / "include"
lib_dir = MNN_INSTALL_DIR / "lib"
bin_dir = MNN_INSTALL_DIR / "bin"

# 拷贝 MNN_INSTALL_DIR / "include" / "llm" 到 MNN_INSTALL_DIR / "include" / "MNN"
llm_dir = MNN_INSTALL_DIR / "include" / "llm"
mnn_dir = MNN_INSTALL_DIR / "include" / "MNN" / "llm"
if llm_dir.exists():
    shutil.copytree(llm_dir, mnn_dir, symlinks=True, dirs_exist_ok=True)
    print(f"Copied {llm_dir} to {mnn_dir}")
    # 删除 llm_dir
    shutil.rmtree(llm_dir)
    
# 拷贝 build_dir / libllm.so 到 MNN_INSTALL_DIR / "lib" ，区分平台
if system == "Windows":
    shutil.copy(build_dir / "llm.dll", bin_dir)
    shutil.copy(build_dir / "llm.lib", lib_dir)
elif system == "Linux":
    shutil.copy(build_dir / "libllm.so", lib_dir)
elif system == "Darwin":
    shutil.copy(build_dir / "libllm.dylib", lib_dir)
elif system == "Android":
    # mkdir
    (lib_dir / args.android_abi).mkdir(parents=True, exist_ok=True)
    shutil.copy(build_dir / "libllm.so", lib_dir)
    # 使用glob模式查找所有.so文件并复制到目标目录
    import glob
    so_files = glob.glob(str(lib_dir / "*.so"))
    for so_file in so_files:
        shutil.copy(so_file, lib_dir / args.android_abi)
    # 删除 lib_dir 下的所有.so文件
    for file in lib_dir.rglob("*.so"):
        if file.parent == lib_dir:  # 只删除直接在lib_dir下的.so文件，不删除子目录中的
            file.unlink()
elif system == "iOS":
    (lib_dir / "arm64").mkdir(parents=True, exist_ok=True)
    shutil.copy(build_dir / "libllm.dylib", lib_dir / "arm64")
    # 使用glob模式查找所有.dylib文件并复制到目标目录
    import glob
    dylib_files = glob.glob(str(lib_dir / "*.dylib"))
    for dylib_file in dylib_files:
        shutil.copy(dylib_file, lib_dir / "arm64")
    # 删除 lib_dir 下的所有.dylib文件
    for file in lib_dir.rglob("*.dylib"):
        if file.parent == lib_dir:  # 只删除直接在lib_dir下的.dylib文件，不删除子目录中的
            file.unlink()

print(f"Include directory: {include_dir} - {'exists' if include_dir.exists() else 'not found'}")
print(f"Library directory: {lib_dir} - {'exists' if lib_dir.exists() else 'not found'}")
print(f"Binary directory: {bin_dir} - {'exists' if bin_dir.exists() else 'not found'}")

if include_dir.exists():
    print("Header files:")
    for header in include_dir.rglob("*.h"):
        print(f"  {header.relative_to(include_dir)}")

if lib_dir.exists():
    print("Library files:")
    for lib_file in lib_dir.rglob("*"):
        if lib_file.is_file() and lib_file.suffix in ['.so', '.dylib', '.dll', '.lib', '.a']:
            print(f"  {lib_file.relative_to(lib_dir)}")

if bin_dir.exists():
    print("Binary files:")
    for bin_file in bin_dir.rglob("*"):
        if bin_file.is_file():
            print(f"  {bin_file.relative_to(bin_dir)}")

print(f"MNN {MNN_VER} build and installation completed!")
print(f"Installation directory: {MNN_INSTALL_DIR}")
