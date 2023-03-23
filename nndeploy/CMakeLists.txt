# this place we can design this way: add test subdirectory the last one
# and set a list varible to store all files needed to be test and finally
# add executable in test subdirectory
if(NNDEPLOY_ENABLE_TEST)
  set(ALL_TEST_SOURCE)
  set(ALL_TEST_OBJECT)
  set(ALL_TEST_TARGET nndeploy_test)
  set(ALL_TEST_DIRECTORY nndeploy)
  set(ALL_TEST_DEPEND_LIBRARY)
  set(ALL_TEST_SYSTEM_LIBRARY)
  set(ALL_TEST_THIRD_PARTY_LIBRARY)
endif()

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

# last line
if(NNDEPLOY_ENABLE_TEST)
  add_subdirectory(source/test)
  unset(ALL_TEST_SOURCE)
  unset(ALL_TEST_OBJECT)
  unset(ALL_TEST_TARGET)
  unset(ALL_TEST_DIRECTORY)
  unset(ALL_TEST_DEPEND_LIBRARY)
  unset(ALL_TEST_SYSTEM_LIBRARY)
  unset(ALL_TEST_THIRD_PARTY_LIBRARY)
endif()