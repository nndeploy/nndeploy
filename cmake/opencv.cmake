
include(ExternalProject)
message(STATUS "opencv")

if (ENABLE_NNDEPLOY_OPENCV STREQUAL "OFF")
elseif (ENABLE_NNDEPLOY_OPENCV STREQUAL "ON")
  message(STATUS "NNTASK_ENABLE_OPENCV: ${NNTASK_ENABLE_OPENCV}")
  find_package(OpenCV REQUIRED)
  # If the package has been found, several variables will
  # be set, you can find the full list with descriptions
  # in the OpenCVConfig.cmake file.
  # Print some message showing some of them
  message(STATUS "OpenCV library status:")
  message(STATUS "    config: ${OpenCV_DIR}") 
  message(STATUS "    version: ${OpenCV_VERSION}")
  message(STATUS "    libraries: ${OpenCV_LIBS}")
  message(STATUS "    include path: ${OpenCV_INCLUDE_DIRS}")
  include_directories(${OpenCV_INCLUDE_DIRS})
  set(NNDEPLOY_THIRD_PARTY_LIBRARY ${NNDEPLOY_THIRD_PARTY_LIBRARY} ${OpenCV_LIBS})
else()
  include_directories(${ENABLE_NNDEPLOY_OPENCV}/include)
  message(STATUS "include_directories(${ENABLE_NNDEPLOY_OPENCV})")
  set(OPENCV "opencv_world480")
  set(tmp_name ${NNDEPLOY_LIB_PREFIX}${OPENCV}${NNDEPLOY_LIB_SUFFIX})
  set(tmp_path ${ENABLE_NNDEPLOY_OPENCV}/lib/x86/Release)
  if (SYSTEM.Windows)
    # set(tmp_name ${OPENCV}${NNDEPLOY_LIB_SUFFIX})
    set(tmp_name ${OPENCV}.lib)
    # set(tmp_path ${ENABLE_NNDEPLOY_OPENCV}//lib//x86//Release//)
  endif()
  set(full_name ${tmp_path}/${tmp_name})
  set(NNDEPLOY_THIRD_PARTY_LIBRARY ${NNDEPLOY_THIRD_PARTY_LIBRARY} ${full_name})
  install(FILES ${full_name} DESTINATION ${NNDEPLOY_INSTALL_PATH})
endif()