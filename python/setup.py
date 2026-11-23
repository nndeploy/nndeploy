
import datetime
import logging
import platform
import shlex
import subprocess
import sys
from glob import glob, iglob
import os
from os import environ, getcwd, path, popen, remove
import shutil


from setuptools import setup, find_packages, Extension 

from packaging.tags import sys_tags
from packaging.requirements import Requirement, InvalidRequirement
from packaging.utils import canonicalize_name
from setuptools.command.build_ext import build_ext as _build_ext
from setuptools.command.install import install as InstallCommandBase

manylinux_tags = [
    "manylinux1_x86_64",
    "manylinux1_i686",
    "manylinux2010_x86_64",
    "manylinux2010_i686",
    "manylinux2014_x86_64",
    "manylinux2014_i686",
    "manylinux2014_aarch64",
    "manylinux2014_armv7l",
    "manylinux2014_ppc64",
    "manylinux2014_ppc64le",
    "manylinux2014_s390x",
    "manylinux_2_28_x86_64",
    "manylinux_2_28_aarch64",
]
is_manylinux = environ.get("AUDITWHEEL_PLAT", None) in manylinux_tags

class InstallCommand(InstallCommandBase):
    def finalize_options(self):
        ret = InstallCommandBase.finalize_options(self)
        self.install_lib = self.install_platlib  # 将库安装到平台特定目录
        return ret

try:
    from wheel.bdist_wheel import bdist_wheel as _bdist_wheel

    class bdist_wheel(_bdist_wheel):
        def finalize_options(self):
            _bdist_wheel.finalize_options(self)
            self.root_is_pure = False
        
        if platform.system() == "Linux":
            def get_tag(self):
                _, _, plat = _bdist_wheel.get_tag(self)
                # Get the right platform tag by querying the linker version
                glibc_major, glibc_minor = popen("ldd --version | head -1").read().split()[-1].split(".")
                """# See https://github.com/mayeut/pep600_compliance/blob/master/
                pep600_compliance/tools/manylinux-policy.json"""
                if glibc_major == "2" and glibc_minor == "17":
                    plat = "manylinux_2_17_x86_64.manylinux2014_x86_64"
                else:  # For manylinux2014 and above, no alias is required
                    plat = f"manylinux_{glibc_major}_{glibc_minor}_x86_64"
                tags = next(sys_tags())
                return (tags.interpreter, tags.abi, plat)

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
       
    # v1 
    # Find all dynamic library files
    all_matches = []
    for ext in extensions:
        pattern = os.path.join(search_path, f"*{ext}*")  # Add * to match versioned suffixes
        matches = glob.glob(pattern)
        all_matches.extend(matches)
    
    if not all_matches:
        raise FileNotFoundError(f"No dynamic library files found in {search_path} directory")
    
    # v2
    # # Find all dynamic library files
    # all_matches = []
    # symlink_info = {}  # Record symlink information
    
    # for ext in extensions:
    #     pattern = os.path.join(search_path, f"*{ext}*")
    #     matches = glob.glob(pattern)
    #     print(f"matches: {matches}")
        
    #     # Analyze each matched file
    #     for match in matches:
    #         if os.path.islink(match):
    #             # If it's a symlink, record the link relationship but don't add to package list
    #             target = os.readlink(match)
    #             symlink_info[os.path.basename(match)] = os.path.basename(target)
    #             print(f"Detected symlink: {os.path.basename(match)} -> {target}")
    #         else:
    #             # Only add real files to package list
    #             all_matches.append(match)
    #             print(f"Added real file: {os.path.basename(match)}")
    
    # if not all_matches:
    #     raise FileNotFoundError(f"No real dynamic library files found in {search_path} directory")
    
    # # Save symlink information to file
    # symlink_info_file = os.path.join(search_path, "_symlink_info.json")
    # if symlink_info:
    #     import json
    #     with open(symlink_info_file, 'w') as f:
    #         json.dump(symlink_info, f, indent=2)
    #     all_matches.append(symlink_info_file)
    #     print(f"Saved symlink info to: {symlink_info_file}")
        
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
                    # First add @loader_path to rpath
                    subprocess.run(["install_name_tool", "-add_rpath", "@loader_path", lib_path], check=False) 
                    try:
                        # Check current library dependencies
                        result = subprocess.run(["otool", "-L", lib_path], 
                                              capture_output=True, text=True)
                        dependencies = result.stdout

                        # Fix all @rpath references to @loader_path
                        for line in dependencies.split('\n'):
                            line = line.strip()
                            if '@rpath/' in line and '.dylib' in line:
                                # Extract library path and name
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
                        
                        # Compatibility handling for macOS versions below 14
                        # Check system version and apply additional fixes
                        import platform as plt
                        macos_version = plt.mac_ver()[0]
                        if macos_version and float('.'.join(macos_version.split('.')[:2])) < 14.0:
                            print(f"Detected macOS {macos_version}, applying compatibility fixes")
                            
                            # Add additional rpath paths for older macOS versions
                            subprocess.run(["install_name_tool", "-add_rpath", "@executable_path", lib_path], check=False)
                            subprocess.run(["install_name_tool", "-add_rpath", ".", lib_path], check=False)
                            
                            # Check and fix library ID
                            lib_name = os.path.basename(lib_path)
                            subprocess.run([
                                "install_name_tool", "-id", 
                                f"@loader_path/{lib_name}", 
                                lib_path
                            ], check=False)
                            
                    except Exception as e:
                        print(f"Warning: macOS dynamic library path fixing failed for {lib_path}: {e}")
                else:
                    subprocess.run(["patchelf", "--set-rpath", "$ORIGIN", lib_path])
            except Exception as e:
                print(f"Warning: Error processing {lib_path}: {e}")
            if platform.system() == "Darwin":
                # Rewriting Mach-O load commands can invalidate the linker signature.
                signature = subprocess.run(["codesign", "--verify", lib_path], capture_output=True)
                if signature.returncode != 0:
                    subprocess.run(["codesign", "--force", "--sign", "-", lib_path], check=True)
                
    # Return list of all dynamic library file names
    return [os.path.basename(match) for match in all_matches]


def get_package_data():
    """Get data files to be packaged"""
    # Directly use the list returned by get_internal_so_path()
    so_files = get_internal_so_path()
    
    # Ensure returning a dictionary with a simple string list as value
    package_data = {
        "nndeploy": so_files + ['e_dit/algorithms/*/requirements.txt']
    }
    
    print(f"package_data: {package_data}")
    return package_data

# Basic dependency packages
install_requires = [
        'cython',  # Cython compilation
        'packaging',  # Package management
        'Pillow',
        'numpy',  # Numerical computation
        'opencv-python>=4.10.0',  # Image processing
    ]

# Add server-related dependencies
server_requires = [
    'modelscope',
    'multiprocess',
    'requests>=2.31.0',  # Request library
    'fastapi>=0.104.0',  # Web framework
    'uvicorn>=0.24.0',  # ASGI server
    'websockets>=11.0',  # WebSocket support
    'python-multipart>=0.0.6',  # File upload support
    'python-dotenv>=1.0,<2',  # Optional server.env loading
    'cryptography>=41.0',  # Encrypted per-request service credentials
    'pydantic>=2.0.0',  # Data validation
    'chardet>=5.2.0',  # Character encoding detection
    'SQLAlchemy>=2.0,<3',
    'alembic>=1.13,<2',
    'prometheus-client>=0.20,<1',
    'threadpoolctl>=3.6,<4',
    'psutil>=5.9',  # Confirm termination of optional compute child processes
]

install_requires.extend(server_requires)

# PyAV is shared with DiT and is already installed for server audio.
e_dit_requires = [
    'torch==2.6.0', 'torchvision==0.21.0', 'diffusers==0.40.0',
    'transformers==5.3.0', 'accelerate==1.13.0', 'peft>=0.12',
    'einops', 'sentencepiece', 'safetensors>=0.8', 'rich', 'imageio-ffmpeg',
    'ffmpeg-python', 'decord', 'moviepy', 'pycocotools', 'pandas', 'scipy', 'ftfy==6.3.1',
]

server_extensions = {
    'server-mcp': ['mcp>=1.30,<2'],
    'server-audio': ['av>=16.1,<20'],
    'server-mysql': ['pymysql>=1.1,<2'],
    'server-postgresql': ['psycopg[binary]>=3.2,<4'],
    'server-redis': ['redis>=5,<8'],
    'server-s3': ['boto3>=1.34,<2'],
    'server-cos': ['cos-python-sdk-v5>=1.9,<2'],
    'server-oss': ['alibabacloud-oss-v2>=1.4,<2'],
    'server-obs': ['esdk-obs-python>=3.25,<4'],
}
for requirements in server_extensions.values():
    install_requires.extend(requirements)


def apply_dependency_overrides(requirements, filename):
    """Replace entire requirements by normalized package name before packaging.

    The file contains PEP 508 requirements and optional pip index options.
    Index options are consumed by pip via -r, not distribution metadata.
    Relative paths are resolved against setup.py, independent of the build cwd.
    Multiple marked requirements for a package replace its default together.
    """
    import re

    if not os.path.isabs(filename):
        filename = os.path.join(os.path.dirname(os.path.abspath(__file__)), filename)

    overrides = []
    names = set()
    with open(filename, encoding='utf-8-sig') as stream:
        for lineno, line in enumerate(stream, 1):
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            line = re.split(r'\s+#', line, maxsplit=1)[0].rstrip()
            if line.startswith('--'):
                option = shlex.split(re.sub(r'^(--[a-z-]+)=', r'\1 ', line))
                if (len(option) == 2 and option[0] in {
                        '--index-url', '--extra-index-url', '--find-links', '--trusted-host'}):
                    continue
                if option == ['--no-index']:
                    continue
            try:
                requirement = Requirement(line)
            except InvalidRequirement as error:
                raise ValueError(
                    f'{filename}:{lineno}: expected a PEP 508 requirement; '
                    'only index options are supported in addition to requirements'
                ) from error
            names.add(canonicalize_name(requirement.name))
            overrides.append(str(requirement))

    retained = [item for item in requirements
                if canonicalize_name(Requirement(item).name) not in names]
    return list(dict.fromkeys(retained + overrides))


def algorithm_requirements_file(algorithm):
    """Select one algorithm without importing model or native dependencies."""
    import re

    if not re.fullmatch(r'[a-z][a-z0-9_]*', algorithm):
        raise ValueError(f'Invalid algorithm name: {algorithm!r}')
    filename = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            'nndeploy', 'e_dit', 'algorithms', algorithm, 'requirements.txt')
    if not os.path.isfile(filename):
        raise FileNotFoundError(f'Algorithm requirements not found: {filename}')
    return filename


algorithm = os.environ.get('NNDEPLOY_DIT_ALGORITHM')
override_file = os.environ.get('NNDEPLOY_REQUIREMENTS_OVERRIDE')
if algorithm:
    if override_file:
        raise ValueError('Set only one of NNDEPLOY_DIT_ALGORITHM and NNDEPLOY_REQUIREMENTS_OVERRIDE')
    override_file = algorithm_requirements_file(algorithm)
if override_file:
    # An explicit algorithm profile opts into DiT, including unmodified defaults.
    install_requires = apply_dependency_overrides(install_requires + e_dit_requires, override_file)

# Server extensions are installed by default. Their old extra names remain valid.
extras_require = {name: [] for name in server_extensions}
extras_require['e_dit'] = [] if override_file else e_dit_requires

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


cmd_classes = {}
if bdist_wheel is not None:
    cmd_classes["bdist_wheel"] = bdist_wheel
cmd_classes["install"] = InstallCommand
# cmd_classes["build_ext"] = build_ext


# Execute copy operation
copy_server_directory()
setup(
    name="nndeploy",
    version="3.0.10",  # Fix version number format
    author="nndeploy team",
    author_email="595961667@qq.com",  # Add email
    description="An Easy-to-Use and High-Performance AI deployment framework",  # Add short description
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
        "Programming Language :: Python :: 3.14",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    license='Apache License 2.0',
    python_requires=">=3.10",
    packages=find_packages(),
    package_dir={"nndeploy": "nndeploy"},
    # package_data=package_data,
    package_data=get_package_data(),
    exclude_package_data={"": [".env", ".env.*", "*.env", "*.env.*",
                               "**/.env", "**/.env.*", "**/*.env", "**/*.env.*"]},
    install_requires=install_requires,
    entry_points={
        'console_scripts': [
            'nndeploy-run-json=nndeploy.dag.run_json:main',
            'nndeploy-app=nndeploy.server.app:main',
            'nndeploy-clean=nndeploy.server.clean:main',
        ],
    },
    extras_require=extras_require,
    cmdclass={"bdist_wheel": bdist_wheel},
    # cmdclass=cmd_classes,
    zip_safe=False,  # Added: Disable zip safe mode
    has_ext_modules=lambda: True,  # Added: Declare that it contains extension modules
    keywords="deep-learning, visual-workflow, ai-agent, easy-to-use, high-performance",
    project_urls={
        "Bug Reports": "https://github.com/nndeploy/nndeploy/issues",
        "Source": "https://github.com/nndeploy/nndeploy",
    },
)
