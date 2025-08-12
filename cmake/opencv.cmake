
include(ExternalProject)

if (ENABLE_NNDEPLOY_OPENCV STREQUAL "OFF")
elseif (ENABLE_NNDEPLOY_OPENCV STREQUAL "ON")
  find_package(OpenCV REQUIRED)
  # If the package has been found, several variables will
  # be set, you can find the full list with descriptions
  # in the OpenCVConfig.cmake file.
  # Print some message showing some of them
  # message(STATUS "OpenCV library status:")
  # message(STATUS "    config: ${OpenCV_DIR}") 
  # message(STATUS "    version: ${OpenCV_VERSION}")
  message(STATUS "    libraries: ${OpenCV_LIBS}")
  # message(STATUS "    include path: ${OpenCV_INCLUDE_DIRS}")
  include_directories(${OpenCV_INCLUDE_DIRS})
  set(NNDEPLOY_THIRD_PARTY_LIBRARY ${NNDEPLOY_THIRD_PARTY_LIBRARY} ${OpenCV_LIBS})
else()
  # message(STATUS "ENABLE_NNDEPLOY_OPENCV: ${ENABLE_NNDEPLOY_OPENCV}")
  # Check if ENABLE_NNDEPLOY_OPENCV is an absolute path
  if(IS_ABSOLUTE ${ENABLE_NNDEPLOY_OPENCV})
    # Absolute path, use directly
    set(OPENCV_ROOT_PATH ${ENABLE_NNDEPLOY_OPENCV})
    message(STATUS "Using absolute path for OpenCV: ${OPENCV_ROOT_PATH}")
  else()
    # Relative path, relative to project root directory
    set(OPENCV_ROOT_PATH ${CMAKE_SOURCE_DIR}/${ENABLE_NNDEPLOY_OPENCV})
    message(STATUS "Using relative path for OpenCV: ${OPENCV_ROOT_PATH}")
    # Update ENABLE_NNDEPLOY_OPENCV to absolute path
    set(ENABLE_NNDEPLOY_OPENCV ${OPENCV_ROOT_PATH})
  endif()
    
  include_directories(${ENABLE_NNDEPLOY_OPENCV}/include)
  include_directories(${ENABLE_NNDEPLOY_OPENCV}/include/opencv4)
  set(LIB_PATH ${ENABLE_NNDEPLOY_OPENCV}/${NNDEPLOY_THIRD_PARTY_LIBRARY_PATH_SUFFIX})
  foreach(LIB ${NNDEPLOY_OPENCV_LIBS})
    if(SYSTEM.Windows)
      set(LIB_NAME ${NNDEPLOY_LIB_PREFIX}${LIB}${NNDEPLOY_OPENCV_VERSION}${NNDEPLOY_LIB_SUFFIX})
    else()
      set(LIB_NAME ${NNDEPLOY_LIB_PREFIX}${LIB}${NNDEPLOY_LIB_SUFFIX})
    endif()
    set(FULL_LIB_NAME ${LIB_PATH}/${LIB_NAME})
    if(IS_SYMLINK ${FULL_LIB_NAME})
      get_filename_component(REAL_LIB_NAME ${FULL_LIB_NAME} REALPATH)
      message(STATUS "Real path of ${FULL_LIB_NAME}: ${REAL_LIB_NAME}")
      set(FULL_LIB_NAME ${REAL_LIB_NAME})
      message(STATUS "Full lib name: ${FULL_LIB_NAME}")
    endif()
    set(NNDEPLOY_THIRD_PARTY_LIBRARY ${NNDEPLOY_THIRD_PARTY_LIBRARY} ${FULL_LIB_NAME})  
  endforeach()
  if(SYSTEM.Windows)
    set(BIN_PATH ${ENABLE_NNDEPLOY_OPENCV}/bin)
    link_directories(${BIN_PATH})
    install(DIRECTORY ${ENABLE_NNDEPLOY_OPENCV}
          DESTINATION ${NNDEPLOY_INSTALL_THIRD_PARTY_PATH}
          PATTERN "opencv" EXCLUDE)
  else()
    install(DIRECTORY ${ENABLE_NNDEPLOY_OPENCV}/include
      DESTINATION ${NNDEPLOY_INSTALL_THIRD_PARTY_PATH}/opencv)
    # 只安装 NNDEPLOY_OPENCV_LIBS 中的库
    foreach(LIB ${NNDEPLOY_OPENCV_LIBS})
      set(LIB_NAME ${NNDEPLOY_LIB_PREFIX}${LIB}*${NNDEPLOY_LIB_SUFFIX})
      # message(${LIB_NAME})
      set(FULL_LIB_PATH ${ENABLE_NNDEPLOY_OPENCV}/${NNDEPLOY_THIRD_PARTY_LIBRARY_PATH_SUFFIX}/${LIB_NAME})
      # 使用file(GLOB)来查找所有以LIB_NAME为前缀的文件
      file(GLOB LIB_FILES "${ENABLE_NNDEPLOY_OPENCV}/${NNDEPLOY_THIRD_PARTY_LIBRARY_PATH_SUFFIX}/${LIB_NAME}*")
      if(LIB_FILES)
        install(FILES ${LIB_FILES}
          DESTINATION ${NNDEPLOY_INSTALL_THIRD_PARTY_PATH}/opencv/${NNDEPLOY_THIRD_PARTY_LIBRARY_PATH_SUFFIX})
      endif()
    endforeach()
  endif()
endif()