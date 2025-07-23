
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
  # message(STATUS "    libraries: ${OpenCV_LIBS}")
  # message(STATUS "    include path: ${OpenCV_INCLUDE_DIRS}")
  include_directories(${OpenCV_INCLUDE_DIRS})
  set(NNDEPLOY_THIRD_PARTY_LIBRARY ${NNDEPLOY_THIRD_PARTY_LIBRARY} ${OpenCV_LIBS})
else()
  include_directories(${ENABLE_NNDEPLOY_OPENCV}/include)
  set(LIB_PATH ${ENABLE_NNDEPLOY_OPENCV}/${NNDEPLOY_THIRD_PARTY_LIBRARY_PATH_SUFFIX})
  foreach(LIB ${NNDEPLOY_OPENCV_LIBS})
    set(LIB_NAME ${NNDEPLOY_LIB_PREFIX}${LIB}${NNDEPLOY_LIB_SUFFIX}${NNDEPLOY_OPENCV_VERSION})
    set(FULL_LIB_NAME ${LIB_PATH}/${LIB_NAME})
    set(NNDEPLOY_THIRD_PARTY_LIBRARY ${NNDEPLOY_THIRD_PARTY_LIBRARY} ${FULL_LIB_NAME})    
  endforeach()
  if(SYSTEM.Windows)
    set(BIN_PATH ${ENABLE_NNDEPLOY_OPENCV}/bin)
    link_directories(${BIN_PATH})
  endif()
  install(DIRECTORY ${ENABLE_NNDEPLOY_OPENCV}
          DESTINATION ${NNDEPLOY_INSTALL_THIRD_PARTY_PATH}
          PATTERN "opencv" EXCLUDE)
endif()