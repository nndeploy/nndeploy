cmake_minimum_required(VERSION 3.10)

# set
set(NNDEPLOY_SOURCE)
set(NNDEPLOY_OBJECT)
set(NNDEPLOY_TARGET nndeploy_base)
set(NNDEPLOY_DIRECTORY nndeploy)
set(NNDEPLOY_DEPEND_LIBRARY)
set(NNDEPLOY_SYSTEM_LIBRARY)
set(NNDEPLOY_THIRD_PARTY_LIBRARY)

include_directories(${ROOT_PATH})

# NNDEPLOY_SOURCE
file(GLOB_RECURSE NNDEPLOY_SOURCE
    "${ROOT_PATH}/nndeploy/include/base/include_c_cpp.h"
    "${ROOT_PATH}/nndeploy/include/base/macro.h"
    "${ROOT_PATH}/nndeploy/include/base/basic.h"
    "${ROOT_PATH}/nndeploy/include/base/type.h"
    "${ROOT_PATH}/nndeploy/include/base/log.h"
    "${ROOT_PATH}/nndeploy/include/base/object.h"
    "${ROOT_PATH}/nndeploy/include/base/openmp.h"
    "${ROOT_PATH}/nndeploy/include/base/thread_pool.h"
    # "${ROOT_PATH}/nndeploy/include/base/file.h"
    # "${ROOT_PATH}/nndeploy/source/base/file.cc"
    # "${ROOT_PATH}/nndeploy/include/base/half.h"
    # "${ROOT_PATH}/nndeploy/source/base/half.cc"
    "${ROOT_PATH}/nndeploy/include/base/time_measurement.h"
    "${ROOT_PATH}/nndeploy/source/base/time_measurement.cc"
    # "${ROOT_PATH}/nndeploy/include/base/shape.h"
    # "${ROOT_PATH}/nndeploy/source/base/shape.cc"
    # "${ROOT_PATH}/nndeploy/include/base/status.h"
    # "${ROOT_PATH}/nndeploy/source/base/status.cc"
    # "${ROOT_PATH}/nndeploy/include/base/string.h"
    # "${ROOT_PATH}/nndeploy/source/base/string.cc"
    # "${ROOT_PATH}/nndeploy/include/base/value.h"
    # "${ROOT_PATH}/nndeploy/source/base/value.cc"
    )
# message(STATUS "NNDEPLOY_SOURCE:${NNDEPLOY_SOURCE}")

# NNDEPLOY_OBJECT

# NNDEPLOY_TARGET
add_library(${NNDEPLOY_TARGET} ${NNDEPLOY_BUILD_TYPE} ${NNDEPLOY_SOURCE} ${NNDEPLOY_OBJECT})

# NNDEPLOY_DIRECTORY
set_property(TARGET ${NNDEPLOY_TARGET} PROPERTY FOLDER ${NNDEPLOY_DIRECTORY})

# NNDEPLOY_DEPEND_LIBRARY

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

