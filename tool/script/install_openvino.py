#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
import shutil
import zipfile
import tarfile
import requests
import platform
import re
from pathlib import Path
import time
import argparse
from tqdm import tqdm
from urllib.parse import urljoin

WORKSPACE_FOLDER = os.path.join(os.path.dirname(__file__), '..', '..')
OV_INSTALL_DIR = os.path.join(WORKSPACE_FOLDER, 'third_party', 'openvino')
OV_PACAGES_URL = "https://storage.openvinotoolkit.org"
EXCLUDE_DIRS = ["pre-release", "nightly", "master"]
OV_FILETREE_URL = "https://storage.openvinotoolkit.org/filetree.json"

def fetch_filetree_json(url):
    """从URL获取filetree.json内容"""
    try:
        response = requests.get(url)
        response.raise_for_status()  # 如果请求失败会抛出异常
        return response.json()
    except requests.exceptions.RequestException as e:
        print(f"获取filetree.json失败: {e}")
        return None
    
def extract_openvino_archive_urls(filetree_data):
    """从filetree.json中提取所有OpenVINO版本archive的下载URL"""
    archives = []
    
    def traverse(node, current_path=""):
        if node["type"] == "directory":
            new_path = f"{current_path}/{node['name']}" if current_path else node['name']
            if "children" in node:
                for child in node["children"]:
                    traverse(child, new_path)
        elif node["type"] == "file":
            # 检查是否在deployment_archives/releases目录下
            if "repositories/openvino/packages" in current_path and is_archive_file(node["name"]) and not any(excl in current_path for excl in EXCLUDE_DIRS):
                file_path = f"{current_path}/{node['name']}"
                download_url = f"{OV_PACAGES_URL}{file_path}".replace("production", "")
                archives.append({
                    "name": node["name"],
                    "path": file_path,
                    "url": download_url,
                    "size": node["size"],
                    "last_modified": node["last modified"]
                })
    
    traverse(filetree_data)
    return archives

def list_available_versions(url=OV_PACAGES_URL):
    filetree_data = fetch_filetree_json(OV_FILETREE_URL)
    archives = extract_openvino_archive_urls(filetree_data)

    available_versions = []

    for archive in archives:
        # 提取版本号，版本号在文件名中以"202x.x"格式出现
        match = re.search(r'(\d{4}\.\d+(\.\d+)*)', archive["url"])
        if match:
            version = match.group(1)
            if version not in available_versions:
                available_versions.append(version)

    return available_versions

def list_available_files(url):
    filetree_data = fetch_filetree_json(OV_FILETREE_URL)
    archives = extract_openvino_archive_urls(filetree_data)
    available_files = []
    for archive in archives:
        if url in archive["url"]:
            available_files.append(archive["url"])

    return available_files

def is_archive_file(file_name):
    return file_name.endswith('.zip') or file_name.endswith('.tar.gz') or file_name.endswith('.tgz')

def is_target_file(file_name, system, arch):
    if system == 'linux':
        system_name = f'ubuntu{args.ubuntu}'
        return (system_name in file_name) and (arch in file_name)
    return (system in file_name) and (arch in file_name)
    
def find_suitable_openvino_url(version, target, arch):
    system_map = {
        'Windows': 'windows',
        'Linux': 'linux',
        'Darwin': 'macos'
    }

    system = system_map.get(target)
    ov_url = f"{version}/{system}"
    available_files = list_available_files(ov_url)
    # print(f"Available files for version {version} on {system}: {available_files}")
    # 选择合适的文件
    target_file = None
    for file in available_files:
        if is_archive_file(file) and is_target_file(file, system, arch):
            target_file = file
            break
    
    if target_file is None:
        print(f"No suitable OpenVINO package found for version {version} on {system} {arch}, available files: {available_files}")
        sys.exit(1)
    print(f"Found suitable file {target_file} for {system} {arch}")

    download_url = target_file

    return download_url

def download_file(url, filename=None):
    if filename is None:
        filename = url.split('/')[-1]

    response = requests.get(url, stream=True)
    if response.status_code != 200:
        print(f"Failed to download file from {url}, status code: {response.status_code}")
        sys.exit(1)

    total_size = int(response.headers.get('content-length', 0))
    if os.path.exists(filename):
        if os.path.getsize(filename) == total_size:
            print(f"File {filename} already exists and is complete, skipping download.")
            return filename
        else:
            print(f"File {filename} already exists but is incomplete, re-downloading.")

    chunk_size = 8192
    with open(filename, 'wb') as file, tqdm(
        desc=filename,
        total=total_size,
        unit='iB',
        unit_scale=True,
        unit_divisor=1024,
    ) as bar:
        for data in response.iter_content(chunk_size=chunk_size):
            size = file.write(data)
            bar.update(size)

    print(f"Downloaded file to {filename}")
    return filename

def extract_archive(archive_path, extract_to='.'):
    extract_dir = None
    if archive_path.endswith('.zip'):
        with zipfile.ZipFile(archive_path, 'r') as zip_ref:
            zip_ref.extractall(extract_to)
            extract_dir = archive_file.rsplit('.', 1)[0]
    elif archive_path.endswith('.tar.gz') or archive_path.endswith('.tgz'):
        with tarfile.open(archive_path, 'r:gz') as tar_ref:
            tar_ref.extractall(extract_to)
            extract_dir = archive_file.rsplit('.', 2)[0] if archive_path.endswith('.tar.gz') else archive_file.rsplit('.', 1)[0]
    else:
        print(f"Unsupported archive format: {archive_path}")
        sys.exit(1)

    return extract_dir
    
def copy_files_with_suffix(src_dir, dest_dir, suffixes):
    if not os.path.exists(dest_dir):
        os.makedirs(dest_dir)

    for root, dirs, files in os.walk(src_dir, followlinks=True):
        for file in files:
            if any(suffix in file for suffix in suffixes):
                shutil.copy2(os.path.join(root, file), dest_dir)

# 解析命令行参数
parser = argparse.ArgumentParser(description='Install OpenVINO precompiled library')

parser.add_argument('--list-versions', action='store_true', help='List available OpenVINO versions and exit')
parser.add_argument('--version', type=str, default='2025.3', help='OpenVINO version to install (default: 2025.3)')
parser.add_argument('--target', type=str, default=None, choices=["Windows", "Linux", "Darwin"], help='Target platform (default: current platform)')
parser.add_argument('--arch', type=str, default='x86_64', choices=['x86_64', 'arm64'], help='Target architecture (default: x86_64)')
parser.add_argument('--ubuntu', type=str, default='20', help='Ubuntu version, only used if target is Linux (default: 20)')

if __name__ == "__main__":
    args = parser.parse_args()
    version = args.version
    target = args.target if args.target else platform.system()
    OV_INSTALL_DIR = OV_INSTALL_DIR + f"_{args.version}"

    if args.list_versions:
        print("Fetching available OpenVINO versions...")
        print(f"Available versions: {', '.join(list_available_versions())}")
        sys.exit(0)

    print(f"Installing OpenVINO version {version} for {target} ({args.arch}) into {OV_INSTALL_DIR}")
    os.makedirs(OV_INSTALL_DIR, exist_ok=True)
    os.chdir(OV_INSTALL_DIR)

    # step 1: 查找合适的下载链接
    ov_url = find_suitable_openvino_url(version, target, args.arch)

    # step 2: 下载文件
    print(f"Downloading OpenVINO ...")
    archive_file = download_file(ov_url)

    # archive_file = "w_openvino_toolkit_windows_2022.3.0.9052.9752fafe8eb_x86_64.zip"
    # step 3: 解压文件夹
    print(f"Extracting {archive_file} ...")
    extract_dir = extract_archive(archive_file)
    print(f"Extracted to {extract_dir}")

    # step 4: 拷贝include文件夹
    include_dir = os.path.join(OV_INSTALL_DIR, 'include')
    if os.path.exists(include_dir):
        shutil.rmtree(include_dir)

    os.makedirs(include_dir, exist_ok=True)
    shutil.copytree(os.path.join(OV_INSTALL_DIR, extract_dir, 'runtime', 'include'), include_dir, symlinks=True, dirs_exist_ok=True)

    print(f"Copied include files to {include_dir}")

    # step 5: 拷贝lib文件夹
    thirdparty_src_dir = os.path.join(OV_INSTALL_DIR, extract_dir, 'runtime', '3rdparty')
    if target == 'Windows':
        if os.path.exists(os.path.join(OV_INSTALL_DIR, 'bin')):
            shutil.rmtree(os.path.join(OV_INSTALL_DIR, 'bin'))
        bin_dir = os.path.join(OV_INSTALL_DIR, 'bin')
        os.makedirs(bin_dir, exist_ok=True)
        
        copy_files_with_suffix(os.path.join(OV_INSTALL_DIR, extract_dir, 'runtime', 'bin'), bin_dir, ['.dll'])
        copy_files_with_suffix(thirdparty_src_dir, bin_dir, ['.dll'])

        print(f"Copied bin files to {bin_dir}")

    lib_dir = os.path.join(OV_INSTALL_DIR, 'lib')
    if os.path.exists(lib_dir):
        shutil.rmtree(lib_dir)
    os.makedirs(lib_dir, exist_ok=True)
    copy_files_with_suffix(os.path.join(OV_INSTALL_DIR, extract_dir, 'runtime', 'lib'), lib_dir, ['.so', '.dylib', '.lib', '.a'])
    copy_files_with_suffix(thirdparty_src_dir, lib_dir, ['.so', '.dylib', '.lib', '.a'])
    print(f"Copied lib files to {lib_dir}")

    # step 6: 清理临时文件
    os.remove(archive_file)
    shutil.rmtree(extract_dir)
    print(f"Removed temporary files")

    print(f"OpenVINO {version} installation completed successfully.")
    
    