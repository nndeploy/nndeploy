cmake_minimum_required(VERSION 3.10)

if(NNDEPLOY_ENABLE_BASE)
  add_subdirectory(source/base)
endif()

if(NNDEPLOY_ENABLE_CRYPTION)
  add_subdirectory(source/cryption)
endif()

if(NNDEPLOY_ENABLE_DEVICE)
  add_subdirectory(source/device)
endif()

if(NNDEPLOY_ENABLE_AUDIO)
  add_subdirectory(source/audio)
endif()

if(NNDEPLOY_ENABLE_CV)
  add_subdirectory(source/cv)
endif()

if(NNDEPLOY_ENABLE_INFERENCE_DEFAULT)
  add_subdirectory(source/ir)
  add_subdirectory(source/interpret)
  add_subdirectory(source/op)
  add_subdirectory(source/forward)
  add_subdirectory(source/optimize)
endif()

if(NNDEPLOY_ENABLE_INFERENCE)
  add_subdirectory(source/inference)
endif()

if(NNDEPLOY_ENABLE_AICOMPILER)
  add_subdirectory(source/aicompiler)
endif()

if(NNDEPLOY_ENABLE_GRAPH)
  add_subdirectory(source/graph)
endif()