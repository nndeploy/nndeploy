#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Linux Build Script for nndeploy
This script automates the build process for nndeploy on Linux platform
Based on GitHub Actions workflow configuration
"""

import os
import sys
import shutil
import subprocess
import platform
from pathlib import Path
import argparse
import multiprocessing

def run_command(cmd, check=True, shell=True):
    """Execute command and handle errors"""
    print(f"Executing: {cmd}")
    try:
        result = subprocess.run(cmd, shell=shell, check=check, capture_output=True, text=True)
        if result.stdout:
            print(result.stdout)
        return result
    except subprocess.CalledProcessError as e:
        print(f"Error executing command: {cmd}")
        print(f"Return code: {e.returncode}")
        if e.stderr:
            print(f"Error output: {e.stderr}")
        if check:
            sys.exit(1)
        return e

def check_system():
    """Check system information"""
    system_info = {
        'platform': platform.platform(),
        'machine': platform.machine(),
        'python_version': platform.python_version(),
        'system': platform.system()
    }
    
    print("System Information:")
    for key, value in system_info.items():
        print(f"  {key}: {value}")
    
    if system_info['system'] != 'Linux':
        print("Warning: This script is designed for Linux systems")
    
    return system_info

def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Build nndeploy on Linux')
    parser.add_argument('--config', 
                       default='config_opencv_ort_mnn_tokenizer.cmake', 
                       type=str, 
                       help='Config file name (config_opencv_ort_mnn_tokenizer.cmake, config_opencv_ort_mnn.cmake, config_opencv_ort.cmake, config_opencv.cmake)')
    parser.add_argument('--build-type', 
                       default='Release', 
                       choices=['Debug', 'Release', 'RelWithDebInfo', 'MinSizeRel'],
                       help='CMake build type')
    parser.add_argument('--jobs', 
                       type=int, 
                       default=multiprocessing.cpu_count(),
                       help='Number of parallel jobs for compilation')
    parser.add_argument('--clean', 
                       action='store_true',
                       help='Clean build directory before building')
    parser.add_argument('--skip-deps', 
                       action='store_true',
                       help='Skip dependency installation')
    parser.add_argument('--skip-third-party', 
                       action='store_true',
                       help='Skip third-party library installation')
    
    return parser.parse_args()

def install_system_dependencies():
    """Install system dependencies using apt-get"""
    print("Installing system dependencies...")
    
    # Update package manager index
    print("Updating package manager index...")
    run_command("sudo apt-get update")
    
    # Install build tools and dependencies
    dependencies = [
        "build-essential",
        "cmake", 
        "make-build",
        "pkg-config",
        "libopencv-dev",
        "protobuf-compiler",
        "libprotobuf-dev",
        "git",
        "wget",
        "curl",
        "unzip",
        "python3-dev",
        "python3-pip"
    ]
    
    print("Installing build tools and dependencies...")
    for dep in dependencies:
        print(f"Installing {dep}...")
        result = run_command(f"sudo apt-get install -y {dep}", check=False)
        if result.returncode != 0:
            print(f"Warning: {dep} installation failed")
    
    print("System dependencies installation completed!")

def install_python_dependencies():
    """Install Python dependencies"""
    print("Installing Python dependencies...")
    
    # Upgrade pip
    print("Upgrading pip...")
    run_command("python3 -m pip install --upgrade pip")
    
    # Install Python dependencies
    python_deps = [
        "pybind11",
        "setuptools", 
        "wheel",
        "twine",
        "requests",
        "pathlib2",
        "cython",
        "numpy"
    ]
    
    for dep in python_deps:
        print(f"Installing {dep}...")
        result = run_command(f"pip3 install {dep}", check=False)
        if result.returncode != 0:
            print(f"Warning: {dep} installation failed")
    
    print("Python dependencies installation completed!")

def install_rust():
    """Install Rust programming language"""
    print("Checking Rust installation status...")
    
    # Check if Rust is already installed
    try:
        result = run_command("rustc --version", check=False)
        if result.returncode == 0:
            print(f"Rust is already installed: {result.stdout.strip()}")
            return True
    except:
        pass
    
    print("Rust not installed, installing...")
    
    # Install Rust using rustup
    print("Downloading and installing Rust...")
    run_command("curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y")
    
    # Source cargo environment
    cargo_env = os.path.expanduser("~/.cargo/env")
    if os.path.exists(cargo_env):
        print("Sourcing cargo environment...")
        run_command(f"source {cargo_env}")
    
    print("Rust installation completed!")
    return True

def install_third_party_libraries():
    """Install third-party libraries"""
    print("Installing third-party libraries...")
    
    # Switch to tool script directory
    script_dir = Path("tool") / "script"
    if not script_dir.exists():
        print(f"Error: Script directory {script_dir} does not exist")
        return False
    
    original_dir = os.getcwd()
    os.chdir(script_dir)
    
    try:
        # Install OpenCV
        print("Installing OpenCV...")
        result = run_command("python3 install_opencv.py", check=False)
        if result.returncode != 0:
            print("Warning: OpenCV installation failed")
        
        # Install ONNX Runtime
        print("Installing ONNX Runtime...")
        result = run_command("python3 install_onnxruntime.py", check=False)
        if result.returncode != 0:
            print("Warning: ONNX Runtime installation failed")
        
        # Build MNN
        print("Building MNN...")
        result = run_command("python3 build_mnn.py", check=False)
        if result.returncode != 0:
            print("Warning: MNN build failed")
            
    finally:
        # Restore original directory
        os.chdir(original_dir)
    
    print("Third-party libraries installation completed!")
    return True

def configure_and_build(config_file, build_type, jobs):
    """Configure CMake and build project"""
    print("Configuring and building project...")
    
    # Create build directory
    build_dir = Path("build")
    if build_dir.exists() and args.clean:
        print(f"Cleaning build directory: {build_dir}")
        shutil.rmtree(build_dir)
    
    build_dir.mkdir(exist_ok=True)
    print(f"Using build directory: {build_dir}")
    
    # Copy configuration file
    config_source = Path("cmake") / config_file
    config_dest = build_dir / "config.cmake"
    
    if config_source.exists():
        shutil.copy2(config_source, config_dest)
        print(f"Copied configuration file: {config_source} -> {config_dest}")
    else:
        print(f"Warning: Configuration file {config_source} does not exist")
    
    # Switch to build directory
    original_dir = os.getcwd()
    os.chdir(build_dir)
    
    try:
        # Configure CMake
        print("Configuring CMake...")
        cmake_cmd = f"cmake -DCMAKE_BUILD_TYPE={build_type} .."
        run_command(cmake_cmd)
        
        # Build project
        print(f"Building project with {jobs} parallel jobs...")
        make_cmd = f"make -j{jobs}"
        run_command(make_cmd)
        
        # Install
        print("Installing...")
        run_command("make install")
        
        # Package
        print("Packaging...")
        run_command("cpack")
        
        # List generated files
        print("Generated files:")
        run_command("ls -la")
        
        print("Compilation, installation and packaging completed")
        
    finally:
        # Restore original directory
        os.chdir(original_dir)
    
    return True

def install_python_package():
    """Install Python package in developer mode and verify"""
    print("Installing Python package in developer mode...")
    
    # Switch to python directory
    python_dir = Path("python")
    if not python_dir.exists():
        print(f"Error: Python directory {python_dir} does not exist")
        return False
    
    original_dir = os.getcwd()
    os.chdir(python_dir)
    
    try:
        # Install in developer mode
        print("Installing Python package in developer mode...")
        run_command("pip3 install -e .")
        
    finally:
        os.chdir(original_dir)
    
    # Verify installation
    print("Verifying Python package installation...")
    verification_script = """
import platform
try:
    import nndeploy
    print(f'✓ Successfully imported nndeploy {nndeploy.__version__}')
    print(f'Platform: {platform.platform()}')
    print(f'Architecture: {platform.machine()}')
    print(f'Python version: {platform.python_version()}')
except ImportError as e:
    print(f'✗ Import failed: {e}')
    exit(1)
"""
    
    result = run_command(f'python3 -c "{verification_script}"', check=False)
    if result.returncode == 0:
        print("Python package developer mode installation and verification completed")
        return True
    else:
        print("Python package verification failed")
        return False

def main():
    """Main build function"""
    print("=" * 60)
    print("nndeploy Linux Build Script")
    print("=" * 60)
    
    # Parse arguments
    global args
    args = parse_arguments()
    
    # Check system
    system_info = check_system()
    
    # Install dependencies
    if not args.skip_deps:
        install_system_dependencies()
        install_python_dependencies()
        install_rust()
    else:
        print("Skipping dependency installation")
    
    # Install third-party libraries
    if not args.skip_third_party:
        install_third_party_libraries()
    else:
        print("Skipping third-party library installation")
    
    # Configure and build
    configure_and_build(args.config, args.build_type, args.jobs)
    
    # Install Python package
    install_python_package()
    
    print("=" * 60)
    print("Build completed successfully!")
    print("=" * 60)

if __name__ == "__main__":
    main()
