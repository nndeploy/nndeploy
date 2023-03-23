# set
set(SOURCE)
set(OBJECT)
set(BINARY nntask_always)
set(DIRECTORY nntask)
set(DEPEND_LIBRARY)
set(SYSTEM_LIBRARY)
set(THIRD_PARTY_LIBRARY)

set(TEST_SOURCE)
set(TEST_OBJECT)

set(DEMO_SOURCE)
set(DEMO_OBJECT)
set(DEMO_BINARY nntask_always_demo)

include_directories(${ROOT_PATH})

# SOURCE
file(GLOB_RECURSE SOURCE
    "${ROOT_PATH}/nntask/include/always/*.h"
    "${ROOT_PATH}/nntask/source/always/*.cc"
    )
file(GLOB_RECURSE TEST_SOURCE
    "${ROOT_PATH}/nntask/include/always/*test.h"
    "${ROOT_PATH}/nntask/source/always/*test.cc"
)
file(GLOB_RECURSE DEMO_SOURCE
    "${ROOT_PATH}/nntask/include/always/*demo.h"
    "${ROOT_PATH}/nntask/source/always/*demo.cc"
)
list(REMOVE_ITEM SOURCE ${TEST_SOURCE})
list(REMOVE_ITEM SOURCE ${DEMO_SOURCE})
message(STATUS "SOURCE: ${SOURCE}")
message(STATUS "DEMO_SOURCE: ${DEMO_SOURCE}")

# OBJECT

# BINARY
add_library(${BINARY} ${BUILD_TYPE} ${SOURCE} ${OBJECT})

# DIRECTORY
set_property(TARGET ${BINARY} PROPERTY FOLDER ${DIRECTORY})

# DEPEND_LIBRARY
list(APPEND DEPEND_LIBRARY nndeploy_base)
target_link_libraries(${BINARY} ${DEPEND_LIBRARY}) 

# SYSTEM_LIBRARY
target_link_libraries(${BINARY} ${SYSTEM_LIBRARY}) 

# THIRD_PARTY_LIBRARY
target_link_libraries(${BINARY} ${THIRD_PARTY_LIBRARY}) 

# install

# test
if(NNTASK_ENABLE_TEST)
  list(APPEND ALL_TEST_SOURCE ${TEST_SOURCE})
  list(APPEND ALL_TEST_OBJECT ${TEST_OBJECT})
  LIST(APPEND ALL_TEST_DEPEND_LIBRARY ${BINARY})
  list(APPEND ALL_TEST_DEPEND_LIBRARY ${DEPEND_LIBRARY})
  list(APPEND ALL_TEST_SYSTEM_LIBRARY ${SYSTEM_LIBRARY})
  list(APPEND ALL_TEST_THIRD_PARTY_LIBRARY ${THIRD_PARTY_LIBRARY})
endif()

# demo
if(NNTASK_ENABLE_DEMO)
  add_executable(${DEMO_BINARY} ${DEMO_SOURCE} ${DEMO_OBJECT})
  set_property(TARGET ${DEMO_BINARY} PROPERTY FOLDER ${DIRECTORY})
  target_link_libraries(${DEMO_BINARY} ${BINARY}) 
  target_link_libraries(${DEMO_BINARY} ${DEPEND_LIBRARY}) 
  target_link_libraries(${DEMO_BINARY} ${SYSTEM_LIBRARY}) 
  target_link_libraries(${DEMO_BINARY} ${THIRD_PARTY_LIBRARY}) 
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

unset(DEMO_SOURCE)
unset(DEMO_OBJECT)
