# set
set(SOURCE)
set(OBJECT)
set(BINARY nndeploy_inference)
set(DIRECTORY nndeploy)
set(DEPEND_LIBRARY)
set(SYSTEM_LIBRARY)
set(THIRD_PARTY_LIBRARY)

set(TEST_SOURCE)
set(TEST_OBJECT)

include_directories(${ROOT_PATH})

# SOURCE
file(GLOB SOURCE
    "${ROOT_PATH}/nndeploy/include/inference/*.h"
    "${ROOT_PATH}/nndeploy/source/inference/*.cc"
)
file(GLOB TEST_SOURCE
    "${ROOT_PATH}/nndeploy/include/inference/*_test.h"
    "${ROOT_PATH}/nndeploy/source/inference/*_test.cc"
)
list(REMOVE_ITEM SOURCE ${TEST_SOURCE})

if (NNDEPLOY_ENABLE_INFERENCE_MNN)
  set(TMP_SOURCE)
  set(TMP_TEST_SOURCE)
  file(GLOB_RECURSE TMP_SOURCE
    "${ROOT_PATH}/nndeploy/include/inference/mnn/*.h"
    "${ROOT_PATH}/nndeploy/source/inference/mnn/*.cc"
  )
  file(GLOB_RECURSE TMP_TEST_SOURCE
    "${ROOT_PATH}/nndeploy/include/inference/mnn/*_test.h"
    "${ROOT_PATH}/nndeploy/source/inference/mnn/*_test.cc"
  )
  list(REMOVE_ITEM TMP_SOURCE ${TMP_TEST_SOURCE})
  list(APPEND SOURCE ${TMP_SOURCE})
  list(APPEND TEST_SOURCE ${TMP_TEST_SOURCE})
  unset(TMP_SOURCE)
  unset(TMP_TEST_SOURCE)
endif()

# OBJECT

# TARGET
add_library(${BINARY} ${BUILD_TYPE} ${SOURCE} ${OBJECT})

# DIRECTORY
set_property(TARGET ${BINARY} PROPERTY FOLDER ${DIRECTORY})

# DEPEND_LIBRARY

# SYSTEM_LIBRARY

# THIRD_PARTY_LIBRARY

# install
# # include(${ROOT_PATH}/cmake/install.cmake)

# test
## repeat?
if(NNDEPLOY_ENABLE_TEST)
  list(APPEND ALL_TEST_SOURCE ${TEST_SOURCE})
  list(APPEND ALL_TEST_OBJECT ${TEST_OBJECT})
  LIST(APPEND ALL_TEST_DEPEND_LIBRARY ${TARGET})
  list(APPEND ALL_TEST_DEPEND_LIBRARY ${DEPEND_LIBRARY})
  list(APPEND ALL_TEST_SYSTEM_LIBRARY ${SYSTEM_LIBRARY})
  list(APPEND ALL_TEST_THIRD_PARTY_LIBRARY ${THIRD_PARTY_LIBRARY})
endif()

# unset
unset(SOURCE)
unset(OBJECT)
unset(BINARY)
unset(DIRECTORY)
unset(DEPEND_LIBRARY)
unset(SYSTEM_LIBRARY)
unset(THIRD_PARTY_LIBRARY)

unset(TEST_SOURCE)
unset(TEST_OBJECT)
