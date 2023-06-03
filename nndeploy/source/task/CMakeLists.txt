# set
set(SOURCE)
set(OBJECT)
set(BINARY nndeploy_task)
set(DIRECTORY nndeploy)
set(DEPEND_LIBRARY)
set(SYSTEM_LIBRARY)
set(THIRD_PARTY_LIBRARY)

include_directories(${ROOT_PATH})

# SOURCE
file(GLOB SOURCE
  "${ROOT_PATH}/nndeploy/source/task/*.h"
  "${ROOT_PATH}/nndeploy/source/task/*.cc"
  )

# OBJECT

# include
if(NNDEPLOY_ENABLE_OPENCV)
  include(${ROOT_PATH}/cmake/opencv.cmake)
endif()
include_directories(${OpenCV_INCLUDE_DIRS})
set(THIRD_PARTY_LIBRARY ${THIRD_PARTY_LIBRARY} ${OpenCV_LIBS})
# include(codec/config.cmake)
include(pre_process/config.cmake)
# include(post_process/config.cmake)
# include(detect/config.cmake)

# TARGET
add_library(${BINARY} ${NNDEPLOY_LIB_TYPE} ${SOURCE} ${OBJECT})

# DIRECTORY
set_property(TARGET ${BINARY} PROPERTY FOLDER ${DIRECTORY})

# DEPEND_LIBRARY
list(APPEND DEPEND_LIBRARY nndeploy_base)
list(APPEND DEPEND_LIBRARY nndeploy_thread)
list(APPEND DEPEND_LIBRARY nndeploy_device)
list(APPEND DEPEND_LIBRARY nndeploy_inference)
target_link_libraries(${BINARY} ${DEPEND_LIBRARY}) 

# SYSTEM_LIBRARY
target_link_libraries(${BINARY} ${SYSTEM_LIBRARY}) 

# THIRD_PARTY_LIBRARY
target_link_libraries(${BINARY} ${THIRD_PARTY_LIBRARY}) 

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

