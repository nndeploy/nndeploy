#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Windows x64 Build Script for nndeploy
This script automates the build process for nndeploy on Windows x64 platform
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
    
    if system_info['system'] != 'Windows':
        print("Warning: This script is designed for Windows systems")
    
    if system_info['machine'] not in ['AMD64', 'x86_64']:
        print("Warning: This script is optimized for x64 architecture")
    
    return system_info

def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Build nndeploy on Windows x64')
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
    parser.add_argument('--generator', 
                       default='Visual Studio 17 2022',
                       help='CMake generator (default: Visual Studio 17 2022)')
    parser.add_argument('--architecture', 
                       default='x64',
                       choices=['x64', 'Win32', 'ARM64'],
                       help='Target architecture (default: x64)')
    
    return parser.parse_args()

def install_system_dependencies():
    """Install system dependencies using Chocolatey or manual installation"""
    print("Installing system dependencies...")
    
    # Check if Chocolatey is installed
    try:
        result = run_command("choco --version", check=False)
        if result.returncode != 0:
            print("Chocolatey not found, installing...")
            run_command('powershell -Command "Set-ExecutionPolicy Bypass -Scope Process -Force; [System.Net.ServicePointManager]::SecurityProtocol = [System.Net.ServicePointManager]::SecurityProtocol -bor 3072; iex ((New-Object System.Net.WebClient).DownloadString(\'https://community.chocolatey.org/install.ps1\'))"')
    except:
        print("Installing Chocolatey...")
        run_command('powershell -Command "Set-ExecutionPolicy Bypass -Scope Process -Force; [System.Net.ServicePointManager]::SecurityProtocol = [System.Net.ServicePointManager]::SecurityProtocol -bor 3072; iex ((New-Object System.Net.WebClient).DownloadString(\'https://community.chocolatey.org/install.ps1\'))"')
    
    # Install build tools and dependencies
    dependencies = [
        "cmake", 
        "git",
        "wget",
        "curl",
        "7zip",
        "python3"
    ]
    
    print("Installing build tools and dependencies...")
    for dep in dependencies:
        print(f"Installing {dep}...")
        result = run_command(f"choco install {dep} -y", check=False)
        if result.returncode != 0:
            print(f"Warning: {dep} installation failed or already installed")
    
    print("System dependencies installation completed!")

def install_python_dependencies():
    """Install Python dependencies"""
    print("Installing Python dependencies...")
    
    # Upgrade pip
    print("Upgrading pip...")
    run_command("python -m pip install --upgrade pip")
    
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
        result = run_command(f"pip install {dep}", check=False)
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
    run_command("curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y", shell=False)
    
    # Add cargo to PATH
    cargo_bin = os.path.expanduser("~/.cargo/bin")
    if os.path.exists(cargo_bin):
        print("Adding cargo to PATH...")
        os.environ['PATH'] = f"{cargo_bin};{os.environ.get('PATH', '')}"
    
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
        result = run_command("python install_opencv.py", check=False)
        if result.returncode != 0:
            print("Warning: OpenCV installation script failed")
        
        # Install ONNX Runtime
        print("Installing ONNX Runtime...")
        result = run_command("python install_onnxruntime.py", check=False)
        if result.returncode != 0:
            print("Warning: ONNX Runtime installation failed")
        
        # Build MNN
        print("Building MNN...")
        result = run_command("python build_mnn.py", check=False)
        if result.returncode != 0:
            print("Warning: MNN build failed")
            
    finally:
        # Restore original directory
        os.chdir(original_dir)
    
    print("Third-party libraries installation completed!")
    return True

def configure_and_build(config_file, build_type, jobs, generator, architecture):
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
        # Configure CMake with Windows x64 specific settings
        print("Configuring CMake...")
        cmake_cmd = f'cmake -G "{generator}" -A {architecture} -DCMAKE_BUILD_TYPE={build_type} ..'
        run_command(cmake_cmd)
        
        # Build project
        print(f"Building project with {jobs} parallel jobs...")
        build_cmd = f"cmake --build . --config {build_type} --parallel {jobs}"
        run_command(build_cmd)
        
        # Install
        print("Installing...")
        install_cmd = f"cmake --install . --config {build_type}"
        run_command(install_cmd)
        
        # Package
        print("Packaging...")
        run_command("cpack")
        
        # List generated files
        print("Generated files:")
        run_command("dir", shell=True)
        
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
        run_command("pip install -e .")
        
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
    print(f'Windows version: {platform.win32_ver()}')
except ImportError as e:
    print(f'✗ Import failed: {e}')
    exit(1)
"""
    
    result = run_command(f'python -c "{verification_script}"', check=False)
    if result.returncode == 0:
        print("Python package developer mode installation and verification completed")
        return True
    else:
        print("Python package verification failed")
        return False

def create_release_package():
    """Create release package similar to GitHub Actions"""
    print("Creating release package...")
    
    build_dir = Path("build")
    if not build_dir.exists():
        print("Error: Build directory does not exist")
        return False
    
    # Find generated zip files
    zip_files = list(build_dir.glob("*.zip"))
    if zip_files:
        print("Found release packages:")
        for zip_file in zip_files:
            print(f"  {zip_file}")
            # Optionally move to a release directory
            release_dir = Path("release")
            release_dir.mkdir(exist_ok=True)
            shutil.copy2(zip_file, release_dir / zip_file.name)
            print(f"  Copied to: {release_dir / zip_file.name}")
    else:
        print("No release packages found")
    
    return True

def main():
    """Main build function"""
    print("=" * 60)
    print("nndeploy Windows x64 Build Script")
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
    configure_and_build(args.config, args.build_type, args.jobs, args.generator, args.architecture)
    
    # Install Python package
    install_python_package()
    
    # Create release package
    create_release_package()
    
    print("=" * 60)
    print("Windows x64 Build completed successfully!")
    print("=" * 60)

if __name__ == "__main__":
    main()
