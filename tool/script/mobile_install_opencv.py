#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
import shutil
import zipfile
import requests
import platform
import argparse
from pathlib import Path

# 设置标准输出编码为UTF-8，解决Windows下中文输出问题
if platform.system() == "Windows":
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

# OpenCV version
OPENCV_VER = "4.10.0"

def get_download_info(target_system):
    """根据目标平台返回对应的下载信息"""
    base_url = f"https://github.com/opencv/opencv/releases/download/{OPENCV_VER}/"
    
    if target_system == "Android":
        filename = f"opencv-{OPENCV_VER}-android-sdk.zip"
        url = base_url + filename
        return url, filename
    elif target_system == "iOS":
        filename = f"opencv-{OPENCV_VER}-ios-framework.zip"
        url = base_url + filename
        return url, filename
    else:
        # 默认下载源码
        filename = f"{OPENCV_VER}.zip"
        url = f"https://github.com/opencv/opencv/archive/refs/tags/{filename}"
        return url, filename

def copy_android_opencv(source_dir, target_dir):
    """复制Android OpenCV文件到third_party目录"""
    print("正在复制Android OpenCV文件...")
    
    # Android OpenCV SDK结构
    android_dirs_to_copy = [
        "sdk/native/jni/include",
        "sdk/native/libs",
        "sdk/native/3rdparty",
        "sdk/java"
    ]
    
    android_files_to_copy = [
        "LICENSE",
        "README.android"
    ]
    
    # 复制目录
    for dir_path in android_dirs_to_copy:
        source_path = source_dir / dir_path
        # 简化目标路径结构
        if "jni/include" in dir_path:
            target_path = target_dir / "include"
        elif "native/libs" in dir_path:
            target_path = target_dir / "lib"
            # target_path = target_dir
        elif "native/3rdparty" in dir_path:
            target_path = target_dir / "3rdparty"
        elif "java" in dir_path:
            target_path = target_dir / "java"
        else:
            target_path = target_dir / Path(dir_path).name
            
        if source_path.exists():
            try:
                if target_path.exists():
                    shutil.rmtree(target_path)
                shutil.copytree(source_path, target_path)
                print(f"已复制目录: {dir_path} -> {target_path.name}")
            except Exception as e:
                print(f"复制目录 {dir_path} 失败: {e}")
        else:
            print(f"警告: 源目录 {dir_path} 不存在")
    
    # 复制文件
    for file_name in android_files_to_copy:
        source_file = source_dir / file_name
        target_file = target_dir / file_name
        
        if source_file.exists():
            try:
                shutil.copy2(source_file, target_file)
                print(f"已复制文件: {file_name}")
            except Exception as e:
                print(f"复制文件 {file_name} 失败: {e}")

def copy_ios_opencv(source_dir, target_dir):
    """复制iOS OpenCV文件到third_party目录"""
    print("正在复制iOS OpenCV文件...")
    
    # iOS OpenCV Framework结构
    ios_dirs_to_copy = [
        "opencv2.framework/Headers",
        "opencv2.framework/Modules"
    ]
    
    ios_files_to_copy = [
        "opencv2.framework/opencv2",
        "opencv2.framework/Info.plist",
        "LICENSE",
        "README.md"
    ]
    
    # 复制Headers到include目录
    headers_source = source_dir / "opencv2.framework/Headers"
    if headers_source.exists():
        headers_target = target_dir / "include"
        try:
            if headers_target.exists():
                shutil.rmtree(headers_target)
            shutil.copytree(headers_source, headers_target)
            print(f"已复制Headers -> include")
        except Exception as e:
            print(f"复制Headers失败: {e}")
    
    # 复制framework目录
    framework_source = source_dir / "opencv2.framework"
    framework_target = target_dir / "framework"
    if framework_source.exists():
        try:
            if framework_target.exists():
                shutil.rmtree(framework_target)
            shutil.copytree(framework_source, framework_target)
            print(f"已复制framework目录")
        except Exception as e:
            print(f"复制framework失败: {e}")
    
    # 复制其他文件
    other_files = ["LICENSE", "README.md"]
    for file_name in other_files:
        source_file = source_dir / file_name
        target_file = target_dir / file_name
        
        if source_file.exists():
            try:
                shutil.copy2(source_file, target_file)
                print(f"已复制文件: {file_name}")
            except Exception as e:
                print(f"复制文件 {file_name} 失败: {e}")

def copy_source_opencv(source_dir, target_dir):
    """复制源码版OpenCV文件到third_party目录"""
    print("正在复制源码版OpenCV文件...")
    
    # 需要复制的目录列表
    dirs_to_copy = [
        "include",
        "modules", 
        "3rdparty",
        "cmake",
        "data"
    ]
    
    # 需要复制的文件列表
    files_to_copy = [
        "CMakeLists.txt",
        "LICENSE",
        "README.md"
    ]
    
    # 复制目录
    for dir_name in dirs_to_copy:
        source_path = source_dir / dir_name
        target_path = target_dir / dir_name
        
        if source_path.exists():
            try:
                if target_path.exists():
                    shutil.rmtree(target_path)
                shutil.copytree(source_path, target_path)
                print(f"已复制目录: {dir_name}")
            except Exception as e:
                print(f"复制目录 {dir_name} 失败: {e}")
        else:
            print(f"警告: 源目录 {dir_name} 不存在")
    
    # 复制文件
    for file_name in files_to_copy:
        source_file = source_dir / file_name
        target_file = target_dir / file_name
        
        if source_file.exists():
            try:
                shutil.copy2(source_file, target_file)
                print(f"已复制文件: {file_name}")
            except Exception as e:
                print(f"复制文件 {file_name} 失败: {e}")
        else:
            print(f"警告: 源文件 {file_name} 不存在")

def main():
    parser = argparse.ArgumentParser(description='下载并安装OpenCV到nndeploy项目')
    parser.add_argument('--system', choices=['Android', 'iOS', 'source'], 
                       default='source', help='目标平台 (默认: source)')
    
    args = parser.parse_args()
    target_system = args.system
    
    # 获取当前执行目录
    WORKSPACE = Path(os.getcwd())
    OPENCV_BUILD_DIR = WORKSPACE / "download"
    
    if target_system == "source":
        OPENCV_DIR = "opencv" + OPENCV_VER
    else:
        OPENCV_DIR = f"opencv{OPENCV_VER}_{target_system}"
    
    OPENCV_INSTALL_DIR = WORKSPACE / "third_party" / OPENCV_DIR
    print(f"OpenCV安装目录: {OPENCV_INSTALL_DIR}")
    print(f"目标平台: {target_system}")
    
    # Create third party library directory
    OPENCV_BUILD_DIR.mkdir(parents=True, exist_ok=True)
    
    # Change to third party library directory
    os.chdir(OPENCV_BUILD_DIR)
    
    # Get download info
    url, filename = get_download_info(target_system)
    
    print(f"正在下载 OpenCV {OPENCV_VER} for {target_system}...")
    print(f"下载链接: {url}")
    
    try:
        response = requests.get(url, stream=True)
        response.raise_for_status()
        
        total_size = int(response.headers.get('content-length', 0))
        downloaded_size = 0
        
        with open(filename, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
                    downloaded_size += len(chunk)
                    if total_size > 0:
                        progress = (downloaded_size / total_size) * 100
                        print(f"\r下载进度: {progress:.1f}%", end='', flush=True)
        print("\n下载完成!")
        
    except requests.RequestException as e:
        print(f"下载失败: {e}")
        print("请检查网络连接或尝试手动下载")
        sys.exit(1)
    
    # Extract source code
    print("正在解压文件...")
    try:
        with zipfile.ZipFile(filename, 'r') as zip_ref:
            zip_ref.extractall(".")
        print("解压完成!")
    except zipfile.BadZipFile as e:
        print(f"解压失败: {e}")
        sys.exit(1)
    
    # 确定解压后的目录名
    if target_system == "Android":
        extracted_dir_name = "OpenCV-android-sdk"
    elif target_system == "iOS":
        extracted_dir_name = "opencv2.framework"  # iOS通常直接解压出framework
        # 但实际可能是包含framework的目录，需要检查
        possible_dirs = ["opencv2.framework", f"opencv-{OPENCV_VER}-ios"]
        extracted_dir_name = None
        for possible_dir in possible_dirs:
            if (OPENCV_BUILD_DIR / possible_dir).exists():
                extracted_dir_name = possible_dir
                break
        if not extracted_dir_name:
            # 查找第一个目录
            for item in OPENCV_BUILD_DIR.iterdir():
                if item.is_dir() and item.name != "__pycache__":
                    extracted_dir_name = item.name
                    break
    else:
        extracted_dir_name = f"opencv-{OPENCV_VER}"
    
    opencv_extracted_dir = OPENCV_BUILD_DIR / extracted_dir_name
    
    if not opencv_extracted_dir.exists():
        print(f"错误: 解压后的目录 {opencv_extracted_dir} 不存在")
        # 列出当前目录内容以便调试
        print("当前目录内容:")
        for item in OPENCV_BUILD_DIR.iterdir():
            print(f"  {item.name}")
        sys.exit(1)
    
    # 按照nndeploy规则复制到third_party目录
    print(f"正在复制到third_party目录: {OPENCV_INSTALL_DIR}")
    
    # 如果目标目录已存在，先删除
    if OPENCV_INSTALL_DIR.exists():
        shutil.rmtree(OPENCV_INSTALL_DIR)
        print(f"已删除现有目录: {OPENCV_INSTALL_DIR}")
    
    # 创建目标目录
    OPENCV_INSTALL_DIR.mkdir(parents=True, exist_ok=True)
    
    # 根据平台复制文件
    if target_system == "Android":
        copy_android_opencv(opencv_extracted_dir, OPENCV_INSTALL_DIR)
    elif target_system == "iOS":
        copy_ios_opencv(opencv_extracted_dir, OPENCV_INSTALL_DIR)
    else:
        copy_source_opencv(opencv_extracted_dir, OPENCV_INSTALL_DIR)
    
    # 清理临时文件
    print("正在清理临时文件...")
    try:
        os.remove(filename)
        if opencv_extracted_dir.exists():
            shutil.rmtree(opencv_extracted_dir)
        print("临时文件清理完成!")
    except Exception as e:
        print(f"清理临时文件时出现警告: {e}")
    
    print(f"\nOpenCV {OPENCV_VER} for {target_system} 安装完成!")
    print(f"安装路径: {OPENCV_INSTALL_DIR}")
    
    if target_system == "Android":
        print("Android使用说明:")
        print(f"  在CMake中设置: -DOpenCV_DIR={OPENCV_INSTALL_DIR}")
        print("  头文件路径: include/")
        print("  库文件路径: lib/arm64-v8a/ (或其他架构)")
    elif target_system == "iOS":
        print("iOS使用说明:")
        print(f"  Framework路径: {OPENCV_INSTALL_DIR}/framework/opencv2.framework")
        print("  在Xcode中添加framework到项目")
        print("  头文件已复制到: include/")
    else:
        print("源码版使用说明:")
        print(f"  在CMake中设置: -DOpenCV_DIR={OPENCV_INSTALL_DIR}")

if __name__ == "__main__":
    main()

