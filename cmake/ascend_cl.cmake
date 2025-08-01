
include(ExternalProject)

if(ENABLE_NNDEPLOY_INFERENCE_ASCEND_CL STREQUAL "OFF")
elseif(ENABLE_NNDEPLOY_INFERENCE_ASCEND_CL STREQUAL "ON")
else()
  set(ENABLE_NNDEPLOY_DEVICE_ASCEND_CL ${ENABLE_NNDEPLOY_INFERENCE_ASCEND_CL})
endif()

if(ENABLE_NNDEPLOY_DEVICE_ASCEND_CL STREQUAL "OFF")
elseif(ENABLE_NNDEPLOY_DEVICE_ASCEND_CL STREQUAL "ON")
else()
  include_directories(${ENABLE_NNDEPLOY_DEVICE_ASCEND_CL}/include)
  include_directories(${ENABLE_NNDEPLOY_DEVICE_ASCEND_CL}/include/aclnn)

  # include_directories(${ENABLE_NNDEPLOY_DEVICE_ASCEND_CL}/include/experiment/platform)
  set(LIB_PATH ${ENABLE_NNDEPLOY_DEVICE_ASCEND_CL}/lib64)
  set(LIBS "ascendcl" "nnopbase" "opapi" "tiling" "platform")

  foreach(LIB ${LIBS})
    set(LIB_NAME ${NNDEPLOY_LIB_PREFIX}${LIB}${NNDEPLOY_LIB_SUFFIX})
    set(FULL_LIB_NAME ${LIB_PATH}/${LIB_NAME})
    set(NNDEPLOY_THIRD_PARTY_LIBRARY ${NNDEPLOY_THIRD_PARTY_LIBRARY} ${FULL_LIB_NAME})
    set(NNDEPLOY_THIRD_PARTY_LIBRARY ${NNDEPLOY_THIRD_PARTY_LIBRARY} ${LIB_PATH}/libtiling_api.a)
  endforeach()

  # ascend c
  nndeploy_option(SOC_VERSION "system on chip type" "Ascend910B4")
  nndeploy_option(RUN_MODE "run mode: npu" "npu")
  set(ASCEND_CANN_PACKAGE_PATH ${ENABLE_NNDEPLOY_DEVICE_ASCEND_CL})
  if(EXISTS ${ENABLE_NNDEPLOY_DEVICE_ASCEND_CL}/tools/tikcpp/ascendc_kernel_cmake)
    set(ASCENDC_CMAKE_DIR ${ENABLE_NNDEPLOY_DEVICE_ASCEND_CL}/tools/tikcpp/ascendc_kernel_cmake)
    include(${ASCENDC_CMAKE_DIR}/ascendc.cmake)
  elseif(EXISTS ${ENABLE_NNDEPLOY_DEVICE_ASCEND_CL}/compiler/tikcpp/ascendc_kernel_cmake)
    set(ASCENDC_CMAKE_DIR ${ENABLE_NNDEPLOY_DEVICE_ASCEND_CL}/compiler/tikcpp/ascendc_kernel_cmake)
    include(${ASCENDC_CMAKE_DIR}/ascendc.cmake)
  elseif(EXISTS ${ENABLE_NNDEPLOY_DEVICE_ASCEND_CL}/ascendc_devkit/tikcpp/samples/cmake)
    set(ASCENDC_CMAKE_DIR ${ENABLE_NNDEPLOY_DEVICE_ASCEND_CL}/ascendc_devkit/tikcpp/samples/cmake)
    include(${ASCENDC_CMAKE_DIR}/ascendc.cmake)
  else()
    message(STATUS "ascendc_kernel_cmake does not exist, please check whether the cann package is installed.")
  endif()

  if(SYSTEM.Windows)
    set(BIN_PATH ${ENABLE_NNDEPLOY_DEVICE_ASCEND_CL}/bin)
    link_directories(${BIN_PATH})
  endif()

  # install(DIRECTORY ${ENABLE_NNDEPLOY_DEVICE_ASCEND_CL} DESTINATION ${NNDEPLOY_INSTALL_THIRD_PARTY_PATH})
endif()