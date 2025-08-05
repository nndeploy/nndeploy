import sys
from setuptools import setup, find_packages
import os
import shutil


try:
    from wheel.bdist_wheel import bdist_wheel as _bdist_wheel

    class bdist_wheel(_bdist_wheel):
        def finalize_options(self):
            _bdist_wheel.finalize_options(self)
            self.root_is_pure = False
            import sys
            python_version = f"{sys.version_info.major}{sys.version_info.minor}"
            self.python_tag = f"cp{python_version}"
            self.abi_tag = f"cp{python_version}"

except ImportError:
    bdist_wheel = None


if sys.version_info < (3, 0):
    sys.exit("Sorry, Python < 3.0 is not supported")


def parse_requirements(fname='requirements.txt', with_version=True):
    """
    Parse the package dependencies listed in a file but strips specific
    versioning information.

    Args:
        fname (str): path to the file
        with_version (bool, default=False): if True include version specs

    Returns:
        List[str]: list of requirements items

    CommandLine:
        python -c "import setup; print(setup.parse_requirements())"
    """
    import re
    import sys
    from os.path import exists
    require_fpath = fname

    def parse_line(line):
        """Parse information from a line in a requirements text file."""
        if line.startswith('-r '):
            # Allow specifying requirements in other files
            target = line.split(' ')[1]
            for info in parse_require_file(target):
                yield info
        else:
            info = {'line': line}
            if line.startswith('-e '):
                info['package'] = line.split('#egg=')[1]
            elif '@git+' in line:
                info['package'] = line
            else:
                # Remove versioning from the package
                pat = '(' + '|'.join(['>=', '==', '>']) + ')'
                parts = re.split(pat, line, maxsplit=1)
                parts = [p.strip() for p in parts]

                info['package'] = parts[0]
                if len(parts) > 1:
                    op, rest = parts[1:]
                    if ';' in rest:
                        # Handle platform specific dependencies
                        # http://setuptools.readthedocs.io/en/latest/setuptools.html#declaring-platform-specific-dependencies
                        version, platform_deps = map(str.strip,
                                                     rest.split(';'))
                        info['platform_deps'] = platform_deps
                    else:
                        version = rest  # NOQA
                    info['version'] = (op, version)
            yield info

    def parse_require_file(fpath):
        with open(fpath, 'r') as f:
            for line in f.readlines():
                line = line.strip()
                if line and not line.startswith('#'):
                    for info in parse_line(line):
                        yield info

    def gen_packages_items():
        if exists(require_fpath):
            for info in parse_require_file(require_fpath):
                parts = [info['package']]
                if with_version and 'version' in info:
                    parts.extend(info['version'])
                if not sys.version.startswith('3.4'):
                    # apparently package_deps are broken in 3.4
                    platform_deps = info.get('platform_deps')
                    if platform_deps is not None:
                        parts.append(';' + platform_deps)
                item = ''.join(parts)
                yield item

    packages = list(gen_packages_items())
    return packages


# Read long description
def read_long_description():
    """Read README.md file as long description"""
    readme_path = "README.md"
    if os.path.exists(readme_path):
        try:
            with open(readme_path, "r", encoding="utf-8") as f:
                return f.read()
        except Exception as e:
            print(f"Warning: Unable to read {readme_path}: {e}")
    
    # If unable to read README, use default description
    return """
You can get everything in nndeploy github main page : [nndeploy](https://github.com/nndeploy/nndeploy)
"""


# def get_internal_so_path():
#     import importlib

#     suffixes = importlib.machinery.EXTENSION_SUFFIXES
#     loader = importlib.machinery.ExtensionFileLoader
#     lazy_loader = importlib.util.LazyLoader.factory(loader)
#     finder = importlib.machinery.FileFinder("nndeploy", (lazy_loader, suffixes))
#     spec = finder.find_spec("_nndeploy_internal")
#     pathname = spec.origin
#     assert os.path.isfile(pathname)
#     return os.path.basename(pathname)


# package_data = {"nndeploy": [get_internal_so_path()]}

def get_internal_so_path():
    import os
    import glob
    import platform
    import subprocess

    # Define search path
    search_path = "nndeploy"
    
    # Set file extensions based on different platforms
    if platform.system() == "Windows":
        extensions = [".dll", ".pyd"]
    elif platform.system() == "Darwin":
        extensions = [".dylib", ".so"]
    else:  # Linux and others
        extensions = [".so"]  # Add .so.* to match versioned dynamic libraries
        
    # Check if directory exists
    if not os.path.exists(search_path):
        raise FileNotFoundError(f"Directory {search_path} does not exist")
        
    # Find all dynamic library files
    all_matches = []
    for ext in extensions:
        pattern = os.path.join(search_path, f"*{ext}*")  # Add * to match versioned suffixes
        matches = glob.glob(pattern)
        all_matches.extend(matches)
    
    if not all_matches:
        raise FileNotFoundError(f"No dynamic library files found in {search_path} directory")
        
    # Print all found dynamic library files
    print(f"Found the following dynamic library files in {search_path} directory:")
    for match in all_matches:
        print(f"  {os.path.basename(match)}")
        
    # Establish dynamic library link relationships on Linux and macOS
    if platform.system() != "Windows":
        for lib_path in all_matches:
            try:
                # Use ldd/otool to view dependencies
                if platform.system() == "Darwin":
                    cmd = ["otool", "-L", lib_path]
                else:
                    cmd = ["ldd", lib_path]
                result = subprocess.run(cmd, capture_output=True, text=True)
                # print(f"\nDependencies of {os.path.basename(lib_path)}:")
                # print(result.stdout)
                
                # Set rpath
                if platform.system() == "Darwin":
                    subprocess.run(["install_name_tool", "-add_rpath", "@loader_path", lib_path]) 
                    try:
                        # Check current library dependencies
                        result = subprocess.run(["otool", "-L", lib_path], 
                                              capture_output=True, text=True)
                        dependencies = result.stdout

                        # Fix all @rpath references to @loader_path
                        for line in dependencies.split('\n'):
                            line = line.strip()
                            if '@rpath/' in line and '.dylib' in line:
                                # Extract library name
                                dylib_path = line.split()[0]
                                dylib_name = os.path.basename(dylib_path)

                                print(f"Fixing rpath for {dylib_name} in {os.path.basename(lib_path)}")

                                # Change @rpath to @loader_path
                                subprocess.run([
                                    "install_name_tool", "-change", 
                                    dylib_path,
                                    f"@loader_path/{dylib_name}", 
                                    lib_path
                                ], check=False)         
                            else:
                                subprocess.run(["patchelf", "--set-rpath", "$ORIGIN", lib_path])
                    except Exception as e:
                        print(f"Warning: Advanced rpath fixing failed for {lib_path}: {e}")
            except Exception as e:
                print(f"Warning: Error processing {lib_path}: {e}")
                
    # Return list of all dynamic library file names
    return [os.path.basename(match) for match in all_matches]


def get_package_data():
    """Get data files to be packaged"""
    # Directly use the list returned by get_internal_so_path()
    so_files = get_internal_so_path()
    
    # Ensure returning a dictionary with a simple string list as value
    package_data = {
        "nndeploy": so_files  # so_files is already a string list
    }
    
    print(f"package_data: {package_data}")
    return package_data

# Basic dependency packages
install_requires = [
    'cython',  # Cython compilation
    'packaging',  # Package management
    # 'setuptools<=68.0.0',  # Setup tools
    'gitpython>=3.1.30',  # Git operations
    'aiofiles>=24.1.0',  # Asynchronous file operations
    'PyYAML>=5.3.1',  # YAML parsing
    'pytest',  # Testing framework
    'jsonschema',  # JSON Schema validation
    'multiprocess',  # Multiprocessing support
    'numpy',  # Numerical computation
    # 'opencv-python>=4.8.0',  # Image processing
]

# Detect if CUDA is available and its version
def get_cuda_version():
    """Detect CUDA version in the system"""
    try:
        # Method 1: Try to get CUDA version through nvidia-smi
        import subprocess
        result = subprocess.run(['nvidia-smi'], capture_output=True, text=True)
        if result.returncode == 0:
            # Extract CUDA version from nvidia-smi output
            lines = result.stdout.split('\n')
            for line in lines:
                if 'CUDA Version:' in line:
                    version = line.split('CUDA Version:')[1].strip().split()[0]
                    # Remove decimal point, e.g., "11.8" -> "118"
                    version_str = version.replace('.','')
                    return version_str
    except:
        pass
    
    try:
        # Method 2: Try to import torch to detect CUDA
        import torch
        if torch.cuda.is_available():
            # Get CUDA version, e.g., "11.8" 
            version = torch.version.cuda
            if version:
                # Remove decimal point, e.g., "118"
                version_str = version.replace('.','')
                return version_str
    except ImportError:
        pass
    except Exception:
        pass
    
    try:
        # Method 3: Check CUDA environment variables
        import os
        cuda_home = os.environ.get('CUDA_HOME') or os.environ.get('CUDA_PATH')
        if cuda_home and os.path.exists(cuda_home):
            # Try to extract version information from path
            version_file = os.path.join(cuda_home, 'version.txt')
            if os.path.exists(version_file):
                with open(version_file, 'r') as f:
                    content = f.read()
                    # Search for version number pattern
                    import re
                    match = re.search(r'CUDA Version (\d+\.\d+)', content)
                    if match:
                        version = match.group(1)
                        version_str = version.replace('.','')
                        return version_str
    except:
        pass
    
    return None

cuda_version = get_cuda_version()

# Add corresponding dependency packages based on CUDA version
if cuda_version:
    print(f"Detected CUDA version: {cuda_version}")
    install_requires.extend([
        'torch>=2.0.0',  # PyTorch (GPU version)
        'torchvision>=0.15.0',  # torchvision (GPU version)
        # 'onnxruntime-gpu>=1.18.0',  # ONNX Runtime GPU version
    ])
else:
    print("CUDA not detected, using CPU version dependencies")
    install_requires.extend([
        'torch>=2.0.0',  # PyTorch (CPU version)
        'torchvision>=0.15.0',  # torchvision (CPU version)
        # 'onnxruntime>=1.18.0',  # ONNX Runtime CPU version
    ])

# Add server-related dependencies
server_requires = [
    'requests>=2.31.0',  # Request library
    'fastapi>=0.104.0',  # Web framework
    'uvicorn>=0.24.0',  # ASGI server
    'websockets>=11.0',  # WebSocket support
    'python-multipart>=0.0.6',  # File upload support
    'pydantic>=2.0.0',  # Data validation
]

install_requires.extend(server_requires)

print(f"Final dependency list: {install_requires}")

# Execute copy, copy ../server directory to nndeploy/ directory
def copy_server_directory():
    """Copy ../server directory to nndeploy/ directory"""
    source_dir = "../server"
    target_dir = "nndeploy/server"
    
    # Items to include
    include_items = {"__pycache__", "frontend"}
    
    if os.path.exists(source_dir):
        # If target directory exists, delete it first
        if os.path.exists(target_dir):
            shutil.rmtree(target_dir)
        
        # Create target directory
        os.makedirs(target_dir, exist_ok=True)
        
        # Manually copy files and directories
        for item in os.listdir(source_dir):
            source_path = os.path.join(source_dir, item)
            target_path = os.path.join(target_dir, item)
            
            if os.path.isdir(source_path):
                # If it's a directory and in the include list
                if item in include_items:
                    shutil.copytree(source_path, target_path)
                    print(f"Copied directory: {item}")
                else:
                    print(f"Skipped directory: {item}")
            else:
                # If it's a Python file, copy directly
                if item.endswith('.py'):
                    shutil.copy2(source_path, target_path)
                    print(f"Copied Python file: {item}")
                else:
                    print(f"Skipped file: {item}")
        
        # Create resources directory under nndeploy/server directory
        resources_dir = os.path.join(target_dir, "resources")
        os.makedirs(resources_dir, exist_ok=True)
        print(f"Created resources directory: {resources_dir}")
        
        print(f"Successfully copied Python files, __pycache__ folder, frontend folder from {source_dir} to {target_dir}")
    else:
        print(f"Source directory {source_dir} does not exist")

# Execute copy operation
copy_server_directory()
setup(
    name="nndeploy",
    version="0.2.11",  # Fix version number format
    author="nndeploy team",
    author_email="595961667@qq.com",  # Add email
    description="Workflow-based Multi-platform AI Deployment Tool",  # Add short description
    long_description=read_long_description(),  # Add long description
    long_description_content_type="text/markdown",  # Specify content type as Markdown
    url="https://github.com/nndeploy/nndeploy",  # Add project URL
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: OS Independent",
        "Programming Language :: C++",
        # "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Programming Language :: Python :: 3.13",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    license='Apache License 2.0',
    python_requires=">=3.10",
    packages=find_packages(),
    package_dir={"nndeploy": "nndeploy"},
    # package_data=package_data,
    package_data=get_package_data(),
    install_requires=install_requires,
    entry_points={
        'console_scripts': [
            'nndeploy-run-json=nndeploy.dag.run_json:main',
            'nndeploy-app=nndeploy.server.app:main',
        ],
    },
    # extras_require={
    #     "all": parse_requirements('../requirements.txt')
    # },
    cmdclass={"bdist_wheel": bdist_wheel},
    keywords="deep-learning, neural-network, model-deployment, inference, ai",
    project_urls={
        "Bug Reports": "https://github.com/nndeploy/nndeploy/issues",
        "Source": "https://github.com/nndeploy/nndeploy",
    },
)
