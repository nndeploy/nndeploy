# this place we can design this way: add test subdirectory the last one 
# and set a list varible to store all files needed to be test and finally 
# add executable in test subdirectory
set(ALL_TEST_SRCS "")

option(NNTASK_ENABLE_SEGMENT "" OFF)
if(NNTASK_ENABLE_SEGMENT)
  add_subdirectory(source/segment)
endif()

if(NNDEPLOY_ENABLE_TEST)
  list(APPEND ALL_TEST_SRCS ${CMAKE_CURRENT_SOURCE_DIR}/test/*_test.cc)
  add_executable(NNDEPLOY_TEST ${ALL_TEST_SRCS})
endif()