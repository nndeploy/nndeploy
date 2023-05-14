# this place we can design this way: add test subdirectory the last one
# and set a list varible to store all files needed to be test and finally
# add executable in test subdirectory
if(NNTASK_ENABLE_TEST)
  set(ALL_TEST_SOURCE)
  set(ALL_TEST_OBJECT)
  set(ALL_TEST_TARGET nntask_test)
  set(ALL_TEST_DIRECTORY nntask)
  set(ALL_TEST_DEPEND_LIBRARY)
  set(ALL_TEST_SYSTEM_LIBRARY)
  set(ALL_TEST_THIRD_PARTY_LIBRARY)
endif()

add_subdirectory(source/common)

nndeploy_option(NNTASK_ENABLE_ALWAYS "NNTASK_ENABLE_ALWAYS" OFF)
if(NNTASK_ENABLE_ALWAYS)
  add_subdirectory(source/always)
endif()

nndeploy_option(NNTASK_ENABLE_ZIP "NNTASK_ENABLE_ZIP" OFF)
if(NNTASK_ENABLE_ZIP)
  add_subdirectory(source/zip)
endif()

nndeploy_option(NNTASK_ENABLE_YOLO "NNTASK_ENABLE_YOLO" OFF)
if(NNTASK_ENABLE_YOLO)
  add_subdirectory(source/yolo)
endif()

# last line
if(NNTASK_ENABLE_TEST)
  add_subdirectory(source/test)
  unset(ALL_TEST_SOURCE)
  unset(ALL_TEST_OBJECT)
  unset(ALL_TEST_TARGET)
  unset(ALL_TEST_DIRECTORY)
  unset(ALL_TEST_DEPEND_LIBRARY)
  unset(ALL_TEST_SYSTEM_LIBRARY)
  unset(ALL_TEST_THIRD_PARTY_LIBRARY)
endif()