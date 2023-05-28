# set
set(SOURCE)
set(OBJECT)
set(BINARY nntask_common)
set(DIRECTORY nntask)
set(DEPEND_LIBRARY)
set(SYSTEM_LIBRARY)
set(THIRD_PARTY_LIBRARY)

set(TEST_SOURCE)
set(TEST_OBJECT)

include_directories(${ROOT_PATH})

# SOURCE
file(GLOB SOURCE
  "${ROOT_PATH}/nntask/source/common/*.h"
  "${ROOT_PATH}/nntask/source/common/*.cc"
  )
file(GLOB TEST_SOURCE
  "${ROOT_PATH}/nntask/source/common/*_test.h"
  "${ROOT_PATH}/nntask/source/common/*_test.cc"
  )
list(REMOVE_ITEM SOURCE ${TEST_SOURCE})

# OBJECT

# include
if(NNTASK_ENABLE_OPENCV)
  include(${ROOT_PATH}/cmake/opencv.cmake)
endif()
include(template_inference_task/config.cmake)
include(pre_process/config.cmake)
include(post_process/config.cmake)

# TARGET
add_library(${BINARY} ${NNDEPLOY_LIB_TYPE} ${SOURCE} ${OBJECT})

# DIRECTORY
set_property(TARGET ${BINARY} PROPERTY FOLDER ${DIRECTORY})

# DEPEND_LIBRARY
list(APPEND DEPEND_LIBRARY nndeploy_base)
list(APPEND DEPEND_LIBRARY nndeploy_device)
list(APPEND DEPEND_LIBRARY nndeploy_inference)
target_link_libraries(${BINARY} ${DEPEND_LIBRARY}) 

# SYSTEM_LIBRARY
target_link_libraries(${BINARY} ${SYSTEM_LIBRARY}) 

# THIRD_PARTY_LIBRARY
target_link_libraries(${BINARY} ${THIRD_PARTY_LIBRARY}) 

# install
install(TARGETS ${BINARY} ${NNDEPLOY_INSTALL_TYPE} DESTINATION ${NNDEPLOY_INSTALL_PATH})

# test
if(NNTASK_ENABLE_TEST)
  set(ALL_TEST_SOURCE ${ALL_TEST_SOURCE} ${TEST_SOURCE} PARENT_SCOPE)
  set(ALL_TEST_OBJECT ${ALL_TEST_OBJECT} ${TEST_OBJECT} PARENT_SCOPE)
  set(ALL_TEST_DEPEND_LIBRARY ${ALL_TEST_DEPEND_LIBRARY} ${TARGET} PARENT_SCOPE)
  set(ALL_TEST_DEPEND_LIBRARY ${ALL_TEST_DEPEND_LIBRARY} ${DEPEND_LIBRARY} PARENT_SCOPE)
  set(ALL_TEST_SYSTEM_LIBRARY ${ALL_TEST_SYSTEM_LIBRARY} ${SYSTEM_LIBRARY} PARENT_SCOPE)
  set(ALL_TEST_THIRD_PARTY_LIBRARY ${ALL_TEST_THIRD_PARTY_LIBRARY} ${THIRD_PARTY_LIBRARY} PARENT_SCOPE)
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
