import sys
from setuptools import setup, find_packages

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
    package_dir={"": "."},
    package_data={"nndeploy": ["nndeploy.cpython-39-x86_64-linux-gnu.so"]},
    cmdclass={"bdist_wheel": bdist_wheel},
)
