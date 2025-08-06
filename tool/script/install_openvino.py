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
import re
import time
import json

# Set standard output encoding to UTF-8 to solve Chinese output issues on Windows
if platform.system() == "Windows":
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

# OpenVINO version
OPENVINO_VER = "2023.1"

# Get current execution directory
WORKSPACE = Path(os.getcwd())
OPENVINO_BUILD_DIR = WORKSPACE / "download"
OPENVINO_DIR = "openvino" + OPENVINO_VER
OPENVINO_INSTALL_DIR = WORKSPACE / "third_party" / OPENVINO_DIR
print(OPENVINO_INSTALL_DIR)

# Create third party library directory
OPENVINO_BUILD_DIR.mkdir(parents=True, exist_ok=True)

# Change to third party library directory
os.chdir(OPENVINO_BUILD_DIR)

def get_available_packages_from_filetree(base_url, max_retries=3):
    """Get available package list from OpenVINO filetree.json"""
    for retry in range(max_retries):
        try:
            print(f"Attempting to get package list from filetree.json (attempt {retry + 1})...")
            
            # Try to get filetree.json
            filetree_url = "https://storage.openvinotoolkit.org/filetree.json"
            response = requests.get(filetree_url, timeout=60, headers={
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
            })
            response.raise_for_status()
            
            filetree = response.json()
            
            # Extract path from base_url
            # base_url like: https://storage.openvinotoolkit.org/repositories/openvino/packages/2023.1/linux/
            path_parts = base_url.replace("https://storage.openvinotoolkit.org/", "").rstrip("/").split("/")
            
            # Navigate through the filetree
            current_node = filetree
            for part in path_parts:
                if part in current_node and isinstance(current_node[part], dict):
                    current_node = current_node[part]
                else:
                    print(f"Path not found in filetree: {'/'.join(path_parts)}")
                    return []
            
            # Extract file names
            packages = []
            if isinstance(current_node, dict):
                for key, value in current_node.items():
                    if isinstance(value, str) and (key.endswith('.tgz') or key.endswith('.zip')):
                        if 'openvino' in key.lower():
                            packages.append(key)
            
            return packages
                    
        except requests.exceptions.Timeout:
            print(f"Request timeout, waiting {2 ** retry} seconds before retry...")
            if retry < max_retries - 1:
                time.sleep(2 ** retry)
        except requests.exceptions.RequestException as e:
            print(f"Network request failed: {e}")
            if retry < max_retries - 1:
                print(f"Waiting {2 ** retry} seconds before retry...")
                time.sleep(2 ** retry)
        except Exception as e:
            print(f"Error occurred while getting package list: {e}")
            if retry < max_retries - 1:
                time.sleep(2 ** retry)
    
    return []

def get_available_packages_from_html(base_url, max_retries=3):
    """Get available package list from HTML parsing (fallback method)"""
    for retry in range(max_retries):
        try:
            print(f"Attempting to get package list from HTML (attempt {retry + 1})...")
            response = requests.get(base_url, timeout=60, headers={
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
            })
            response.raise_for_status()
            
            # Use regex to match download links
            # Match links like href="l_openvino_toolkit_..."
            pattern = r'href="([^"]*l_openvino_toolkit_[^"]*\.(?:tgz|zip))"'
            matches = re.findall(pattern, response.text)
            
            if matches:
                return [match for match in matches if not match.startswith('http')]
            else:
                print("OpenVINO package links not found in HTML content, trying other patterns...")
                # Try other possible matching patterns
                pattern2 = r'href="([^"]*openvino[^"]*\.(?:tgz|zip))"'
                matches2 = re.findall(pattern2, response.text)
                if matches2:
                    return [match for match in matches2 if not match.startswith('http')]
                    
        except requests.exceptions.Timeout:
            print(f"Request timeout, waiting {2 ** retry} seconds before retry...")
            if retry < max_retries - 1:
                time.sleep(2 ** retry)
        except requests.exceptions.RequestException as e:
            print(f"Network request failed: {e}")
            if retry < max_retries - 1:
                print(f"Waiting {2 ** retry} seconds before retry...")
                time.sleep(2 ** retry)
        except Exception as e:
            print(f"Error occurred while getting package list: {e}")
            if retry < max_retries - 1:
                time.sleep(2 ** retry)
    
    return []

def get_available_packages(base_url, max_retries=3):
    """Get available package list using multiple methods"""
    # First try filetree.json method
    packages = get_available_packages_from_filetree(base_url, max_retries)
    
    # If filetree method fails, try HTML parsing
    if not packages:
        print("Filetree method failed, trying HTML parsing...")
        packages = get_available_packages_from_html(base_url, max_retries)
    
    return packages

def find_best_package(packages, system, machine):
    """Find the most suitable package for the current platform from available packages"""
    if system == "Linux" and machine == "x86_64":
        # Prefer ubuntu20 or ubuntu22 versions
        for package in packages:
            if "ubuntu20" in package and "x86_64" in package:
                return package
        for package in packages:
            if "ubuntu22" in package and "x86_64" in package:
                return package
        # If ubuntu versions not found, look for generic linux versions
        for package in packages:
            if "linux" in package and "x86_64" in package:
                return package
    elif system == "Linux" and machine in ["aarch64", "arm64"]:
        for package in packages:
            if "arm" in package:
                return package
    elif system == "Windows" and machine in ["AMD64", "x86_64"]:
        for package in packages:
            if "windows" in package:
                return package
    elif system == "Darwin":  # macOS
        if machine == "x86_64":
            for package in packages:
                if "macos" in package and "arm64" not in package:
                    return package
        elif machine in ["arm64", "aarch64"]:
            for package in packages:
                if "macos" in package and "arm64" in package:
                    return package
    
    return None

def get_fallback_packages():
    """Return known package names when unable to get package list from official repository"""
    system = platform.system()
    machine = platform.machine()
    
    fallback_packages = {
        "Linux": {
            "x86_64": [
                "l_openvino_toolkit_ubuntu20_2023.1.0.12185.47b736f63ed_x86_64.tgz",
                "l_openvino_toolkit_ubuntu22_2023.1.0.12185.47b736f63ed_x86_64.tgz",
                "l_openvino_toolkit_centos7_2023.1.0.12185.47b736f63ed_x86_64.tgz"
            ]
        },
        "Windows": {
            "AMD64": [
                "w_openvino_toolkit_windows_2023.1.0.12185.47b736f63ed_x86_64.zip"
            ]
        },
        "Darwin": {
            "x86_64": [
                "m_openvino_toolkit_macos_10_15_2023.1.0.12185.47b736f63ed_x86_64.tgz"
            ],
            "arm64": [
                "m_openvino_toolkit_macos_11_0_2023.1.0.12185.47b736f63ed_arm64.tgz"
            ]
        }
    }
    
    return fallback_packages.get(system, {}).get(machine, [])

# Determine download URL and filename based on platform
def get_download_info():
    """Return corresponding download information based on current platform"""
    system = platform.system()
    machine = platform.machine()
    
    # Determine base URL based on platform
    if system == "Windows":
        base_url = f"https://storage.openvinotoolkit.org/repositories/openvino/packages/{OPENVINO_VER}/windows/"
    elif system == "Linux":
        base_url = f"https://storage.openvinotoolkit.org/repositories/openvino/packages/{OPENVINO_VER}/linux/"
    elif system == "Darwin":  # macOS
        base_url = f"https://storage.openvinotoolkit.org/repositories/openvino/packages/{OPENVINO_VER}/macos/"
    else:
        raise ValueError(f"Unsupported operating system: {system}")
    
    print(f"Querying available packages: {base_url}")
    
    # Get available package list
    packages = get_available_packages(base_url)
    
    # If unable to get package list, use fallback package names
    if not packages:
        print("Unable to get package list from official repository, trying known package names...")
        packages = get_fallback_packages()
        if not packages:
            raise Exception(f"Unable to get package list from {base_url} and no fallback packages available")
    
    print(f"Found {len(packages)} available packages:")
    for pkg in packages:
        print(f"  {pkg}")
    
    # Find the most suitable package
    filename = find_best_package(packages, system, machine)
    if not filename:
        # If no best matching package found, use the first available package
        if packages:
            filename = packages[0]
            print(f"Warning: No perfect match found, using first available package: {filename}")
        else:
            raise ValueError(f"No suitable OpenVINO package found for {system} {machine}")
    
    url = base_url + filename
    print(f"Selected package: {filename}")
    
    return url, filename

# Get download information
url, filename = get_download_info()

# Check if file already exists
if os.path.exists(filename):
    print(f"File {filename} already exists, skipping download")
else:
    # Download OpenVINO precompiled version
    print(f"Downloading OpenVINO {OPENVINO_VER}...")
    print(f"Download URL: {url}")
    
    try:
        response = requests.get(url, stream=True, timeout=60, headers={
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        })
        response.raise_for_status()
        
        total_size = int(response.headers.get('content-length', 0))
        downloaded = 0
        
        with open(filename, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
                    downloaded += len(chunk)
                    if total_size > 0:
                        percent = (downloaded / total_size) * 100
                        print(f"\rDownload progress: {percent:.1f}% ({downloaded}/{total_size} bytes)", end='', flush=True)
        
        print(f"\nDownload completed: {filename}")
        
    except requests.exceptions.RequestException as e:
        raise Exception(f"Download failed: {e}")

# Extract file
print("Extracting file...")
if filename.endswith('.zip'):
    with zipfile.ZipFile(filename, 'r') as zip_ref:
        zip_ref.extractall(".")
elif filename.endswith('.tgz') or filename.endswith('.tar.gz'):
    with tarfile.open(filename, 'r:gz') as tar_ref:
        tar_ref.extractall(".")
else:
    raise ValueError(f"Unsupported file format: {filename}")

# Find extracted directory
extracted_dirs = [d for d in os.listdir('.') if os.path.isdir(d) and 'openvino' in d and d != 'openvino']
if not extracted_dirs:
    raise Exception("Extracted OpenVINO directory not found")

extracted_dir = extracted_dirs[0]
print(f"Found extracted directory: {extracted_dir}")

# Rename directory
if os.path.exists("openvino"):
    shutil.rmtree("openvino")

try:
    os.rename(extracted_dir, "openvino")
except PermissionError:
    shutil.move(extracted_dir, "openvino")

# Install to target directory
print(f"Installing to target directory: {OPENVINO_INSTALL_DIR}")
if OPENVINO_INSTALL_DIR.exists():
    shutil.rmtree(OPENVINO_INSTALL_DIR)

# Copy entire directory
shutil.copytree("openvino", OPENVINO_INSTALL_DIR)

# Verify installation
print("Verifying installation:")
print(f"Header directory: {OPENVINO_INSTALL_DIR}/runtime/include - {'exists' if (OPENVINO_INSTALL_DIR / 'runtime' / 'include').exists() else 'not found'}")
print(f"Library directory: {OPENVINO_INSTALL_DIR}/runtime/lib/intel64 - {'exists' if (OPENVINO_INSTALL_DIR / 'runtime' / 'lib' / 'intel64').exists() else 'not found'}")
print(f"Binary directory: {OPENVINO_INSTALL_DIR}/runtime/bin/intel64 - {'exists' if (OPENVINO_INSTALL_DIR / 'runtime' / 'bin' / 'intel64').exists() else 'not found'}")

include_dir = OPENVINO_INSTALL_DIR / "runtime" / "include"
lib_dir = OPENVINO_INSTALL_DIR / "runtime" / "lib" / "intel64"
bin_dir = OPENVINO_INSTALL_DIR / "runtime" / "bin" / "intel64"

if platform.system() == "Windows":
    lib_dir = OPENVINO_INSTALL_DIR / "runtime" / "lib" / "intel64" / "Release"
    bin_dir = OPENVINO_INSTALL_DIR / "runtime" / "bin" / "intel64" / "Release"
    
target_include_dir = OPENVINO_INSTALL_DIR / "include"
target_lib_dir = OPENVINO_INSTALL_DIR / "lib"
target_bin_dir = OPENVINO_INSTALL_DIR / "bin"

# Create simplified directory structure for easier CMake discovery
print("Creating simplified directory structure...")
target_include_dir.parent.mkdir(parents=True, exist_ok=True)
target_lib_dir.mkdir(parents=True, exist_ok=True)
target_bin_dir.mkdir(parents=True, exist_ok=True)

# Copy header files
if include_dir.exists():
    print(f"Copying header files from {include_dir} to {target_include_dir}")
    if target_include_dir.exists():
        shutil.rmtree(target_include_dir)
    shutil.copytree(include_dir, target_include_dir)

# Copy library files
if lib_dir.exists():
    print(f"Copying library files from {lib_dir} to {target_lib_dir}")
    for lib_file in lib_dir.iterdir():
        if lib_file.is_file():
            shutil.copy2(lib_file, target_lib_dir)

# Copy binary files
if bin_dir.exists():
    print(f"Copying binary files from {bin_dir} to {target_bin_dir}")
    for bin_file in bin_dir.iterdir():
        if bin_file.is_file():
            shutil.copy2(bin_file, target_bin_dir)


if target_include_dir.exists():
    print("Target header directory:")
    for header in target_include_dir.glob("**/*.h"):
        print(f"  {header.relative_to(target_include_dir)}")

if target_lib_dir.exists():
    print("Target library directory:")
    for lib_file in target_lib_dir.iterdir():
        if lib_file.is_file():
            print(f"  {lib_file.name}")

if target_bin_dir.exists():
    print("Target binary directory:")
    for bin_file in target_bin_dir.iterdir():
        if bin_file.is_file():
            print(f"  {bin_file.name}")

print(f"OpenVINO {OPENVINO_VER} installation completed!")
