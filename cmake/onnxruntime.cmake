
include(ExternalProject)

if (ENABLE_NNDEPLOY_INFERENCE_ONNXRUNTIME STREQUAL "OFF")
elseif (ENABLE_NNDEPLOY_INFERENCE_ONNXRUNTIME STREQUAL "ON")
else()
  # message(STATUS "ENABLE_NNDEPLOY_INFERENCE_ONNXRUNTIME: ${ENABLE_NNDEPLOY_INFERENCE_ONNXRUNTIME}")
  # 判断ENABLE_NNDEPLOY_INFERENCE_ONNXRUNTIME是否为绝对路径
  if(IS_ABSOLUTE ${ENABLE_NNDEPLOY_INFERENCE_ONNXRUNTIME})
    # Absolute path, use directly
    set(ONNXRUNTIME_ROOT_PATH ${ENABLE_NNDEPLOY_INFERENCE_ONNXRUNTIME})
    message(STATUS "Using absolute path for ONNX Runtime: ${ONNXRUNTIME_ROOT_PATH}")
  else()
    # Relative path, relative to project root directory
    set(ONNXRUNTIME_ROOT_PATH ${CMAKE_SOURCE_DIR}/${ENABLE_NNDEPLOY_INFERENCE_ONNXRUNTIME})
    message(STATUS "Using relative path for ONNX Runtime: ${ONNXRUNTIME_ROOT_PATH}")
    # Update ENABLE_NNDEPLOY_INFERENCE_ONNXRUNTIME to absolute path
    set(ENABLE_NNDEPLOY_INFERENCE_ONNXRUNTIME ${ONNXRUNTIME_ROOT_PATH})
  endif()

  include_directories(${ENABLE_NNDEPLOY_INFERENCE_ONNXRUNTIME}/include)
  set(LIB_PATH ${ENABLE_NNDEPLOY_INFERENCE_ONNXRUNTIME}/${NNDEPLOY_THIRD_PARTY_LIBRARY_PATH_SUFFIX})
  set(LIBS "onnxruntime")
  foreach(LIB ${LIBS})
    set(LIB_NAME ${NNDEPLOY_LIB_PREFIX}${LIB}${NNDEPLOY_LIB_SUFFIX})
    set(FULL_LIB_NAME ${LIB_PATH}/${LIB_NAME})
    set(NNDEPLOY_THIRD_PARTY_LIBRARY ${NNDEPLOY_THIRD_PARTY_LIBRARY} ${FULL_LIB_NAME})    
  endforeach()
  if(SYSTEM.Windows)
    set(BIN_PATH ${ENABLE_NNDEPLOY_INFERENCE_ONNXRUNTIME}/bin)
    link_directories(${BIN_PATH})
  endif()
  install(DIRECTORY ${ENABLE_NNDEPLOY_INFERENCE_ONNXRUNTIME} DESTINATION ${NNDEPLOY_INSTALL_THIRD_PARTY_PATH})
endif()