# set
set(SOURCE)
set(OBJECT)
set(TARGET nndeploy_base)
set(DIRECTORY nndeploy)
set(DEPEND_LIBRARY)
set(SYSTEM_LIBRARY)
set(THIRD_PARTY_LIBRARY)

set(TEST_SOURCE)
set(TEST_OBJECT)

include_directories(${ROOT_PATH})

# SOURCE
file(GLOB_RECURSE SOURCE
    "${ROOT_PATH}/nndeploy/include/base/*.h"
    "${ROOT_PATH}/nndeploy/source/base/*.cc"
)
file(GLOB_RECURSE TEST_SOURCE
    "${ROOT_PATH}/nndeploy/include/base/*_test.h"
    "${ROOT_PATH}/nndeploy/source/base/*_test.cc"
)
list(REMOVE_ITEM SOURCE "${ROOT_PATH}/nndeploy/include/base/file.h")
list(REMOVE_ITEM SOURCE "${ROOT_PATH}/nndeploy/source/base/file.cc")
list(REMOVE_ITEM SOURCE "${ROOT_PATH}/nndeploy/include/base/string.h")
list(REMOVE_ITEM SOURCE "${ROOT_PATH}/nndeploy/source/base/string.cc")
list(REMOVE_ITEM SOURCE ${TEST_SOURCE})

# OBJECT

# TARGET
add_library(${TARGET} ${BUILD_TYPE} ${SOURCE} ${OBJECT})

# DIRECTORY
set_property(TARGET ${TARGET} PROPERTY FOLDER ${DIRECTORY})

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
unset(TARGET)
unset(DIRECTORY)
unset(DEPEND_LIBRARY)
unset(SYSTEM_LIBRARY)
unset(THIRD_PARTY_LIBRARY)

unset(TEST_SOURCE)
unset(TEST_OBJECT)
