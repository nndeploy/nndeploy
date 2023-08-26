# TODO

## 2023.08.08
+ 添加address_sanitizer - 完成
+ 完成git submodules 
  + git submodule add git@github.com:gflags/gflags.git
  + git submodule add git@github.com:Alwaysssssss/nndeploy_resource.git
  + git submodule update --init --recursive
  + git submodule update --remote third_party/gflags
  + git submodule update --init --recursive
  + git submodule sync
  + https://gist.github.com/myusuf3/7f645819ded92bda6677
  + (https://blog.csdn.net/Java0258/article/details/108532507?spm=1001.2101.3001.6650.2&utm_medium=distribute.pc_relevant.none-task-blog-2%7Edefault%7ECTRLIST%7ERate-2-108532507-blog-120493968.235%5Ev38%5Epc_relevant_default_base&depth_1-utm_source=distribute.pc_relevant.none-task-blog-2%7Edefault%7ECTRLIST%7ERate-2-108532507-blog-120493968.235%5Ev38%5Epc_relevant_default_base)

[submodule "resource/nndeploy_resource"]
	path = resource/nndeploy_resource
	url = git@github.com:Alwaysssssss/nndeploy_resource.git

## 2023.08.10
+ cmake警告 - CMake Warning (dev) at cmake/cuda.cmake:29 (find_package):
  Policy CMP0146 is not set: The FindCUDA module is removed.  Run "cmake
  --help-policy CMP0146" for policy details.  Use the cmake_policy command to
  set the policy and suppress this warning.