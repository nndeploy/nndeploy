message(STATUS "plugin/track")

# set
set(PLUGIN_SOURCE)
set(PLUGIN_OBJECT)
set(PLUGIN_BINARY nndeploy_plugin_track)

# SOURCE
file(GLOB PLUGIN_SOURCE
  "${PLUGIN_ROOT_PATH}/include/nndeploy/track/*.h"
  "${PLUGIN_ROOT_PATH}/source/nndeploy/track/*.cc"
)

if(ENABLE_NNDEPLOY_PLUGIN_TRACK_FAIRMOT)
  file(GLOB_RECURSE FAIRMOT_SOURCE
    "${PLUGIN_ROOT_PATH}/include/nndeploy/track/fairmot/*.h"
    "${PLUGIN_ROOT_PATH}/source/nndeploy/track/fairmot/*.cc"
  )
  set(PLUGIN_SOURCE ${PLUGIN_SOURCE} ${FAIRMOT_SOURCE})
  message(STATUS "  + FAIRMOT track backend")
else()
  message(STATUS "  - FAIRMOT track backend (disabled)")
endif()

if(ENABLE_NNDEPLOY_PLUGIN_TRACK_BYTETRACK)
  file(GLOB_RECURSE BYTETRACK_SOURCE
    "${PLUGIN_ROOT_PATH}/include/nndeploy/track/bytetrack/*.h"
    "${PLUGIN_ROOT_PATH}/source/nndeploy/track/bytetrack/*.cc"
  )
  set(PLUGIN_SOURCE ${PLUGIN_SOURCE} ${BYTETRACK_SOURCE})
  message(STATUS "  + ByteTrack track backend")
else()
  message(STATUS "  - ByteTrack track backend (disabled)")
endif()

if(ENABLE_NNDEPLOY_PLUGIN_TRACK_BOTSORT)
  file(GLOB_RECURSE BOTSORT_SOURCE
    "${PLUGIN_ROOT_PATH}/include/nndeploy/track/botsort/*.h"
    "${PLUGIN_ROOT_PATH}/source/nndeploy/track/botsort/*.cc"
  )
  set(PLUGIN_SOURCE ${PLUGIN_SOURCE} ${BOTSORT_SOURCE})
  # BotSORT depends on ByteTrack
  if(NOT ENABLE_NNDEPLOY_PLUGIN_TRACK_BYTETRACK)
    set(ENABLE_NNDEPLOY_PLUGIN_TRACK_BYTETRACK ON)
  endif()
  # BotSORT: cv::estimateAffinePartial2D ← opencv_video, cv::RANSAC ← opencv_calib3d
  if(NOT "${ENABLE_NNDEPLOY_OPENCV}" STREQUAL "OFF")
    set(_botsort_ocv_path "${ENABLE_NNDEPLOY_OPENCV}/${NNDEPLOY_THIRD_PARTY_LIBRARY_PATH_SUFFIX}")
    foreach(_mod video calib3d)
      set(_lib "${_botsort_ocv_path}/${NNDEPLOY_LIB_PREFIX}opencv_${_mod}${NNDEPLOY_LIB_SUFFIX}")
      if(NOT EXISTS "${_lib}")
        set(_lib "${_botsort_ocv_path}/libopencv_${_mod}.a")
      endif()
      if(EXISTS "${_lib}")
        target_link_libraries(${PLUGIN_BINARY} PRIVATE "${_lib}")
      endif()
    endforeach()
    unset(_botsort_ocv_path)
    unset(_lib)
  endif()
  message(STATUS "  + BotSORT track backend")
else()
  message(STATUS "  - BotSORT track backend (disabled)")
endif()

if(ENABLE_NNDEPLOY_PLUGIN_TRACK_BOXMOT)
  file(GLOB_RECURSE BOXMOT_SOURCE
    "${PLUGIN_ROOT_PATH}/include/nndeploy/track/boxmot/*.h"
    "${PLUGIN_ROOT_PATH}/source/nndeploy/track/boxmot/*.cc"
  )
  set(PLUGIN_SOURCE ${PLUGIN_SOURCE} ${BOXMOT_SOURCE})
  # BoxMot native C++ trackers
  include(${CMAKE_CURRENT_SOURCE_DIR}/cmake/boxmot.cmake)
  message(STATUS "  + BoxMot unified track backend (native C++)")
else()
  message(STATUS "  - BoxMot unified track backend (disabled)")
endif()

# # TARGET
add_library(${PLUGIN_BINARY} ${NNDEPLOY_LIB_TYPE} ${PLUGIN_SOURCE} ${PLUGIN_OBJECT})

# # DIRECTORY
set_property(TARGET ${PLUGIN_BINARY} PROPERTY FOLDER ${NNDEPLOY_PLUGIN_DIRECTORY})

# # DEPEND_LIBRARY
target_link_libraries(${PLUGIN_BINARY} PRIVATE ${NNDEPLOY_DEPEND_LIBRARY})

# # SYSTEM_LIBRARY
target_link_libraries(${PLUGIN_BINARY} PRIVATE ${NNDEPLOY_SYSTEM_LIBRARY})

# # THIRD_PARTY_LIBRARY
target_link_libraries(${PLUGIN_BINARY} PRIVATE ${NNDEPLOY_THIRD_PARTY_LIBRARY})

# # NNDEPLOY_FRAMEWORK_BINARY
target_link_libraries(${PLUGIN_BINARY} PRIVATE ${NNDEPLOY_FRAMEWORK_BINARY})
target_link_libraries(${PLUGIN_BINARY} PRIVATE nndeploy_plugin_preprocess)
target_link_libraries(${PLUGIN_BINARY} PRIVATE nndeploy_plugin_infer)

# # NNDEPLOY_PLUGIN_THIRD_PARTY_LIBRARY
target_link_libraries(${PLUGIN_BINARY} PRIVATE ${NNDEPLOY_PLUGIN_THIRD_PARTY_LIBRARY})

# # BoxMot native tracker libraries
if(ENABLE_NNDEPLOY_PLUGIN_TRACK_BOXMOT AND TARGET boxmot_tracker_base)
  target_link_libraries(${PLUGIN_BINARY} PRIVATE boxmot_tracker_base)
  foreach(_tracker bytetrack botsort ocsort sfsort occluboost)
    if(TARGET ${_tracker}_core)
      target_link_libraries(${PLUGIN_BINARY} PRIVATE ${_tracker}_core)
    endif()
  endforeach()
  target_link_libraries(${PLUGIN_BINARY} PRIVATE Eigen3::Eigen)
endif()

# # install
if(SYSTEM_Windows)
  install(TARGETS ${PLUGIN_BINARY} ${NNDEPLOY_INSTALL_TYPE} DESTINATION ${NNDEPLOY_INSTALL_PATH})
  install(DIRECTORY ${PLUGIN_ROOT_PATH}/include/nndeploy/track DESTINATION ${NNDEPLOY_INSTALL_INCLUDE_PATH}/nndeploy)
else()
  install(TARGETS ${PLUGIN_BINARY} ${NNDEPLOY_INSTALL_TYPE} DESTINATION ${NNDEPLOY_INSTALL_LIB_PATH})
  install(DIRECTORY ${PLUGIN_ROOT_PATH}/include/nndeploy/track DESTINATION ${NNDEPLOY_INSTALL_INCLUDE_PATH}/nndeploy)
endif()

# appedn list
set(NNDEPLOY_PLUGIN_LIST ${NNDEPLOY_PLUGIN_LIST} ${PLUGIN_BINARY})

# unset
unset(PLUGIN_SOURCE)
unset(PLUGIN_OBJECT)
unset(PLUGIN_BINARY)
