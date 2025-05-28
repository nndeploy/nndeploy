
include(ExternalProject)
include(cmake/util.cmake)

if (ENABLE_NNDEPLOY_DEVICE_OPENCL STREQUAL "OFF")
elseif (ENABLE_NNDEPLOY_DEVICE_OPENCL STREQUAL "ON")
  set(OPENCL_ROOT ${PROJECT_SOURCE_DIR}/third_party/OpenCL-Headers)
  include_directories(${OPENCL_ROOT})
else()
endif()