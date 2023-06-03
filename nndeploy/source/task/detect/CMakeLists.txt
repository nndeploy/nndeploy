# set
set(SOURCE)
set(OBJECT)
set(BINARY nndeploy_task_detect)
set(DIRECTORY nndeploy)
set(DEPEND_LIBRARY)
set(SYSTEM_LIBRARY)
set(THIRD_PARTY_LIBRARY)

include_directories(${ROOT_PATH})

# SOURCE
file(GLOB_RECURSE SOURCE
  "${ROOT_PATH}/nndeploy/source/task/detect/*.h"
  "${ROOT_PATH}/nndeploy/source/task/detect/*.cc"
  )

# OBJECT

if(NNTASK_ENABLE_OPENCV)
  include(${ROOT_PATH}/cmake/opencv.cmake)
endif()
include(opencv/config.cmake)

# BINARY
add_library(${BINARY} ${NNDEPLOY_LIB_TYPE} ${SOURCE} ${OBJECT})

# DIRECTORY
set_property(TARGET ${BINARY} PROPERTY FOLDER ${DIRECTORY})

# DEPEND_LIBRARY
list(APPEND DEPEND_LIBRARY nndeploy_base)
list(APPEND DEPEND_LIBRARY nndeploy_thread)
list(APPEND DEPEND_LIBRARY nndeploy_device)
list(APPEND DEPEND_LIBRARY nndeploy_inference)
list(APPEND DEPEND_LIBRARY nndeploy_task)
target_link_libraries(${BINARY} ${DEPEND_LIBRARY}) 

# SYSTEM_LIBRARY
target_link_libraries(${BINARY} ${SYSTEM_LIBRARY}) 

# THIRD_PARTY_LIBRARY
message(STATUS "opencv libs: ${OpenCV_LIBS}")
message(STATUS "THIRD_PARTY_LIBRARY: ${THIRD_PARTY_LIBRARY}")
target_link_libraries(${BINARY} ${THIRD_PARTY_LIBRARY}) 
target_link_libraries(${BINARY} ${OpenCV_LIBS}) 

# install
install(TARGETS ${BINARY} ${NNDEPLOY_INSTALL_TYPE} DESTINATION ${NNDEPLOY_INSTALL_PATH})

# unset
unset(SOURCE)
unset(OBJECT)
unset(BINARY)
unset(DIRECTORY)
unset(DEPEND_LIBRARY)
unset(SYSTEM_LIBRARY)
unset(THIRD_PARTY_LIBRARY)
