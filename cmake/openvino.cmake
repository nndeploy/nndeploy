
include(ExternalProject)

if (ENABLE_NNDEPLOY_INFERENCE_OPENVINO STREQUAL "OFF")
elseif (ENABLE_NNDEPLOY_INFERENCE_OPENVINO STREQUAL "ON")
  # 使用系统安装的OpenVINO (通过 apt install openvino-2023.1.0 安装)
  find_package(OpenVINO REQUIRED COMPONENTS Runtime)
  
  # 设置包含目录
  include_directories(${OpenVINO_INCLUDE_DIRS})
  
  # 链接OpenVINO库
  set(NNDEPLOY_THIRD_PARTY_LIBRARY ${NNDEPLOY_THIRD_PARTY_LIBRARY} openvino::runtime)
  
  message(STATUS "Using system installed OpenVINO")
else()
  if(IS_ABSOLUTE ${ENABLE_NNDEPLOY_INFERENCE_OPENVINO})
    # Absolute path, use directly
    set(ENABLE_NNDEPLOY_INFERENCE_OPENVINO ${ENABLE_NNDEPLOY_INFERENCE_OPENVINO})
    message(STATUS "Using absolute path for OpenVINO: ${ENABLE_NNDEPLOY_INFERENCE_OPENVINO}")
  else()
    # Relative path, relative to project root directory
    set(ENABLE_NNDEPLOY_INFERENCE_OPENVINO ${CMAKE_SOURCE_DIR}/${ENABLE_NNDEPLOY_INFERENCE_OPENVINO})
    message(STATUS "Using relative path for OpenVINO: ${ENABLE_NNDEPLOY_INFERENCE_OPENVINO}")
    # Update ENABLE_NNDEPLOY_INFERENCE_OPENVINO to absolute path
    set(ENABLE_NNDEPLOY_INFERENCE_OPENVINO ${ENABLE_NNDEPLOY_INFERENCE_OPENVINO})
  endif()

  include_directories(${ENABLE_NNDEPLOY_INFERENCE_OPENVINO}/include)
  set(LIB_PATH ${ENABLE_NNDEPLOY_INFERENCE_OPENVINO}/${NNDEPLOY_THIRD_PARTY_LIBRARY_PATH_SUFFIX})
  set(LIBS "openvino")
  foreach(LIB ${LIBS})
    set(LIB_NAME ${NNDEPLOY_LIB_PREFIX}${LIB}${NNDEPLOY_LIB_SUFFIX})
    set(FULL_LIB_NAME ${LIB_PATH}/${LIB_NAME})
    if(IS_SYMLINK ${FULL_LIB_NAME})
      get_filename_component(REAL_LIB_NAME ${FULL_LIB_NAME} REALPATH)
      # message(STATUS "Real path of ${FULL_LIB_NAME}: ${REAL_LIB_NAME}")
      set(FULL_LIB_NAME ${REAL_LIB_NAME})
      # message(STATUS "Full lib name: ${FULL_LIB_NAME}")
    endif()
    set(NNDEPLOY_THIRD_PARTY_LIBRARY ${NNDEPLOY_THIRD_PARTY_LIBRARY} ${FULL_LIB_NAME})    
  endforeach()
  if(SYSTEM_Windows)
    set(BIN_PATH ${ENABLE_NNDEPLOY_INFERENCE_OPENVINO}/bin)
    link_directories(${BIN_PATH})
    file(GLOB_RECURSE SET_BIN_PATH ${BIN_PATH}/*.dll*)
    foreach(SET_BIN_PATH ${SET_BIN_PATH})
      file(COPY ${SET_BIN_PATH} DESTINATION ${EXECUTABLE_OUTPUT_PATH}/${CMAKE_BUILD_TYPE})
    endforeach()
  endif()
  install(DIRECTORY ${ENABLE_NNDEPLOY_INFERENCE_OPENVINO} DESTINATION ${NNDEPLOY_INSTALL_THIRD_PARTY_PATH}) 
endif()