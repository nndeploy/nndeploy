import sys
from setuptools import setup, find_packages
import os


try:
    from wheel.bdist_wheel import bdist_wheel as _bdist_wheel

    class bdist_wheel(_bdist_wheel):
        def finalize_options(self):
            _bdist_wheel.finalize_options(self)
            self.root_is_pure = False


except ImportError:
    bdist_wheel = None

if sys.version_info < (3, 0):
    sys.exit("Sorry, Python < 3.0 is not supported")


def get_internal_so_path():
    import importlib

    suffixes = importlib.machinery.EXTENSION_SUFFIXES
    loader = importlib.machinery.ExtensionFileLoader
    lazy_loader = importlib.util.LazyLoader.factory(loader)
    finder = importlib.machinery.FileFinder("nndeploy", (lazy_loader, suffixes))
    spec = finder.find_spec("_nndeploy_internal")
    pathname = spec.origin
    assert os.path.isfile(pathname)
    return os.path.basename(pathname)

package_data = {"nndeploy": [get_internal_so_path()]}

setup(
    name="nndeploy",
    version="2.0.0.0",
    author="nndeploy team",
    classifiers=[
        "Programming Language :: C++",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
    ],

    python_requires=">=3.5",
    packages=find_packages(),
    package_dir={"nndeploy": "nndeploy"},
    package_data=package_data,
    cmdclass={"bdist_wheel": bdist_wheel},
)
