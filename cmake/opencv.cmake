
include(ExternalProject)

if (ENABLE_NNDEPLOY_OPENCV STREQUAL "ON")
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
  set(TMP_THIRD_PARTY_LIBRARY ${TMP_THIRD_PARTY_LIBRARY} ${OpenCV_LIBS})
else()
  include_directories(${NNTASK_ENABLE_OPENCV}/include)
  message(STATUS "include_directories(${NNTASK_ENABLE_OPENCV})")
  set(OPENCV "OPENCV")
  set(tmp_name ${NNDEPLOY_LIB_PREFIX}${OPENCV}${NNDEPLOY_LIB_SUFFIX})
  if (SYSTEM.Windows)
    set(tmp_name ${OPENCV}${NNDEPLOY_LIB_SUFFIX})
  endif()
  set(tmp_path ${NNTASK_ENABLE_OPENCV}/lib/x86/Release)
  set(full_name ${tmp_path}/${tmp_name})
  set(TMP_THIRD_PARTY_LIBRARY ${TMP_THIRD_PARTY_LIBRARY} ${full_name})
  install(FILES ${full_name} DESTINATION ${NNDEPLOY_INSTALL_PATH})
endif()