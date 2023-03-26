# set
set(TMP_SOURCE)
set(TMP_OBJECT)
set(TMP_BINARY nndeploy_inference_mnn)
set(TMP_DIRECTORY nndeploy)
set(TMP_DEPEND_LIBRARY)
set(TMP_SYSTEM_LIBRARY)
set(TMP_THIRD_PARTY_LIBRARY)

set(TMP_TEST_SOURCE)
set(TMP_TEST_OBJECT)

include_directories(${ROOT_PATH})

# TMP_SOURCE
file(GLOB_RECURSE TMP_SOURCE
  "${ROOT_PATH}/nndeploy/include/inference/mnn/*.h"
  "${ROOT_PATH}/nndeploy/source/inference/mnn/*.cc"
)
file(GLOB_RECURSE TMP_TEST_SOURCE
  "${ROOT_PATH}/nndeploy/include/inference/mnn/*_test.h"
  "${ROOT_PATH}/nndeploy/source/inference/mnn/*_test.cc"
)
list(REMOVE_ITEM TMP_SOURCE ${TMP_TEST_SOURCE})
list(APPEND TMP_SOURCE ${TMP_SOURCE})

# TMP_OBJECT

# macro or function
include(${ROOT_PATH}/cmake/mnn.cmake)
set(NEED_LIBRARY "ALL")
mnn_macro(NNDEPLOY_ENABLE_INFERENCE_MNN NEED_LIBRARY TMP_SYSTEM_LIBRARY TMP_THIRD_PARTY_LIBRARY)
unset(NEED_LIBRARY)

# TARGET
# add_library(${TMP_BINARY} ${BUILD_TYPE} ${TMP_SOURCE} ${TMP_OBJECT})

# TMP_DIRECTORY
# set_property(TARGET ${TMP_BINARY} PROPERTY FOLDER ${TMP_DIRECTORY})

# TMP_DEPEND_LIBRARY
# target_link_libraries(${TMP_BINARY} ${TMP_DEPEND_LIBRARY}) 

# TMP_DEPEND_LIBRARY
# target_link_libraries(${TMP_BINARY} ${TMP_DEPEND_LIBRARY}) 

# TMP_THIRD_PARTY_LIBRARY
# target_link_libraries(${TMP_BINARY} ${TMP_THIRD_PARTY_LIBRARY}) 

# install

# test
## repeat?
if(NNDEPLOY_ENABLE_TEST)
  list(APPEND TEST_SOURCE ${TMP_TEST_SOURCE})
  # list(APPEND TEST_OBJECT ${TMP_TEST_OBJECT})
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
