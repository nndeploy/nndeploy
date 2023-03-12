cmake_minimum_required(VERSION 3.10)

# set
set(NNDEPLOY_SOURCE)
set(NNDEPLOY_OBJECT)
set(NNDEPLOY_TARGET nndeploy_device)
set(NNDEPLOY_DIRECTORY nndeploy)
set(NNDEPLOY_DEPEND_LIBRARY)
set(NNDEPLOY_SYSTEM_LIBRARY)
set(NNDEPLOY_THIRD_PARTY_LIBRARY)

include_directories(${ROOT_PATH})

# NNDEPLOY_SOURCE
file(GLOB_RECURSE NNDEPLOY_SOURCE
    "${ROOT_PATH}/nndeploy/include/device/architecture.h"
    "${ROOT_PATH}/nndeploy/source/device/architecture.cc"
    "${ROOT_PATH}/nndeploy/include/device/device.h"
    "${ROOT_PATH}/nndeploy/source/device/device.cc"
    "${ROOT_PATH}/nndeploy/include/device/buffer_pool.h"
    "${ROOT_PATH}/nndeploy/include/device/buffer_pool.cc"
    "${ROOT_PATH}/nndeploy/include/device/buffer.h"
    "${ROOT_PATH}/nndeploy/source/device/buffer.cc"
    )
# message(STATUS "NNDEPLOY_SOURCE:${NNDEPLOY_SOURCE}")

# NNDEPLOY_OBJECT

# NNDEPLOY_TARGET
add_library(${NNDEPLOY_TARGET} ${NNDEPLOY_BUILD_TYPE} ${NNDEPLOY_SOURCE} ${NNDEPLOY_OBJECT})

# NNDEPLOY_DIRECTORY
set_property(TARGET ${NNDEPLOY_TARGET} PROPERTY FOLDER ${NNDEPLOY_DIRECTORY})

# NNDEPLOY_DEPEND_LIBRARY
target_link_libraries(${NNDEPLOY_TARGET} ${NNTASK_TARGET} nndeploy_base) 

# NNDEPLOY_SYSTEM_LIBRARY

# NNDEPLOY_THIRD_PARTY_LIBRARY

# install
## include(${ROOT_PATH}/cmake/install.cmake)

# unset
unset(NNDEPLOY_SOURCE)
unset(NNDEPLOY_OBJECT)
unset(NNDEPLOY_TARGET)
unset(NNDEPLOY_DIRECTORY)
unset(NNDEPLOY_DEPEND_LIBRARY)
unset(NNDEPLOY_SYSTEM_LIBRARY)
unset(NNDEPLOY_THIRD_PARTY_LIBRARY)

