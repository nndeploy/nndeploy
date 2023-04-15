# set
set(TMP_SOURCE)
set(TMP_OBJECT)
set(TMP_BINARY nndeploy_inference_tensor_rt)
set(TMP_DIRECTORY nndeploy)
set(TMP_DEPEND_LIBRARY)
set(TMP_SYSTEM_LIBRARY)
set(TMP_THIRD_PARTY_LIBRARY)

set(TMP_TEST_SOURCE)
set(TMP_TEST_OBJECT)

include_directories(${ROOT_PATH})

# TMP_SOURCE
file(GLOB_RECURSE TMP_SOURCE
  "${ROOT_PATH}/nndeploy/source/inference/tensor_rt/*.h"
  "${ROOT_PATH}/nndeploy/source/inference/tensor_rt/*.cc"
  )
file(GLOB_RECURSE TMP_TEST_SOURCE
  "${ROOT_PATH}/nndeploy/source/inference/tensor_rt/*_test.h"
  "${ROOT_PATH}/nndeploy/source/inference/tensor_rt/*_test.cc"
  )
list(REMOVE_ITEM TMP_SOURCE ${TMP_TEST_SOURCE})
list(APPEND SOURCE ${TMP_SOURCE})

# TMP_OBJECT

# include
include(${ROOT_PATH}/cmake/tensor_rt.cmake)

# TARGET
# add_library(${TMP_BINARY} ${NNDEPLOY_LIB_TYPE} ${TMP_SOURCE} ${TMP_OBJECT})

# TMP_DIRECTORY
# set_property(TARGET ${TMP_BINARY} PROPERTY FOLDER ${TMP_DIRECTORY})

# TMP_DEPEND_LIBRARY
list(APPEND DEPEND_LIBRARY ${TMP_DEPEND_LIBRARY}) 

# TMP_SYSTEM_LIBRARY
list(APPEND SYSTEM_LIBRARY ${TMP_SYSTEM_LIBRARY}) 

# TMP_THIRD_PARTY_LIBRARY
list(APPEND THIRD_PARTY_LIBRARY ${TMP_THIRD_PARTY_LIBRARY}) 

# install

# testcl
if(NNDEPLOY_ENABLE_TEST)
  list(APPEND TEST_SOURCE ${TMP_TEST_SOURCE})
  list(APPEND TEST_OBJECT ${TMP_TEST_OBJECT})
endif()

# unset
unset(TMP_SOURCE)
unset(TMP_OBJECT)
unset(TMP_BINARY)
unset(TMP_DIRECTORY)
unset(TMP_DEPEND_LIBRARY)
unset(TMP_SYSTEM_LIBRARY)
unset(TMP_THIRD_PARTY_LIBRARY)

unset(TMP_TEST_SOURCE)
unset(TMP_TEST_OBJECT)