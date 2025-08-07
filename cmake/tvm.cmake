
include(ExternalProject)

if (ENABLE_NNDEPLOY_INFERENCE_TVM STREQUAL "OFF")
elseif (ENABLE_NNDEPLOY_INFERENCE_TVM STREQUAL "ON")
else()
  if(IS_ABSOLUTE ${ENABLE_NNDEPLOY_INFERENCE_TVM})
    # Absolute path, use directly
    set(ENABLE_NNDEPLOY_INFERENCE_TVM ${ENABLE_NNDEPLOY_INFERENCE_TVM})
    message(STATUS "Using absolute path for TVM: ${ENABLE_NNDEPLOY_INFERENCE_TVM}")
  else()
    # Relative path, relative to project root directory
    set(ENABLE_NNDEPLOY_INFERENCE_TVM ${CMAKE_SOURCE_DIR}/${ENABLE_NNDEPLOY_INFERENCE_TVM})
    message(STATUS "Using relative path for TVM: ${ENABLE_NNDEPLOY_INFERENCE_TVM}")
    # Update ENABLE_NNDEPLOY_INFERENCE_TVM to absolute path
    set(ENABLE_NNDEPLOY_INFERENCE_TVM ${ENABLE_NNDEPLOY_INFERENCE_TVM})
  endif()

  include_directories(${ENABLE_NNDEPLOY_INFERENCE_TVM}/include)
  include_directories(${ENABLE_NNDEPLOY_INFERENCE_TVM}/3rdparty/cnpy)
  include_directories(${ENABLE_NNDEPLOY_INFERENCE_TVM}/3rdparty/dlpack/include)
  include_directories(${ENABLE_NNDEPLOY_INFERENCE_TVM}/3rdparty/dmlc-core/include)

  set(LIB_PATH ${ENABLE_NNDEPLOY_INFERENCE_TVM}/lib)
  set(LIBS "tvm")
  foreach(LIB ${LIBS})
    set(LIB_NAME ${NNDEPLOY_LIB_PREFIX}${LIB}${NNDEPLOY_LIB_SUFFIX})
    set(FULL_LIB_NAME ${LIB_PATH}/${LIB_NAME})
    set(NNDEPLOY_THIRD_PARTY_LIBRARY ${NNDEPLOY_THIRD_PARTY_LIBRARY} ${FULL_LIB_NAME})
  endforeach()
  if(SYSTEM.Windows)
    set(BIN_PATH ${ENABLE_NNDEPLOY_INFERENCE_TVM}/bin)
    link_directories(${BIN_PATH})
  endif()
  install(DIRECTORY ${ENABLE_NNDEPLOY_INFERENCE_TVM} DESTINATION ${NNDEPLOY_INSTALL_THIRD_PARTY_PATH})
endif()