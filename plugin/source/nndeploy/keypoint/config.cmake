message(STATUS "plugin/keypoint")

# set
set(PLUGIN_SOURCE)
set(PLUGIN_OBJECT)
set(PLUGIN_BINARY nndeploy_plugin_keypoint)

# SOURCE
file(GLOB PLUGIN_SOURCE
  "${PLUGIN_ROOT_PATH}/include/nndeploy/keypoint/*.h"
  "${PLUGIN_ROOT_PATH}/source/nndeploy/keypoint/*.cc"
)

if(ENABLE_NNDEPLOY_PLUGIN_KEYPOINT_YOLO_POSE)
  file(GLOB_RECURSE YOLO_POSE_SOURCE
    "${PLUGIN_ROOT_PATH}/include/nndeploy/keypoint/yolo_pose/*.h"
    "${PLUGIN_ROOT_PATH}/source/nndeploy/keypoint/yolo_pose/*.cc"
  )
  set(PLUGIN_SOURCE ${PLUGIN_SOURCE} ${YOLO_POSE_SOURCE})
  message(STATUS "  + YOLO-Pose keypoint backend")
else()
  message(STATUS "  - YOLO-Pose keypoint backend (disabled)")
endif()

if(ENABLE_NNDEPLOY_PLUGIN_KEYPOINT_RF_DETR_POSE)
  file(GLOB_RECURSE RF_DETR_POSE_SOURCE
    "${PLUGIN_ROOT_PATH}/include/nndeploy/keypoint/rf_detr_pose/*.h"
    "${PLUGIN_ROOT_PATH}/source/nndeploy/keypoint/rf_detr_pose/*.cc"
  )
  set(PLUGIN_SOURCE ${PLUGIN_SOURCE} ${RF_DETR_POSE_SOURCE})
  message(STATUS "  + RF-DETR-Pose keypoint backend")
else()
  message(STATUS "  - RF-DETR-Pose keypoint backend (disabled)")
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

# # install
if(SYSTEM_Windows)
  install(TARGETS ${PLUGIN_BINARY} ${NNDEPLOY_INSTALL_TYPE} DESTINATION ${NNDEPLOY_INSTALL_PATH})
  install(DIRECTORY ${PLUGIN_ROOT_PATH}/include/nndeploy/keypoint DESTINATION ${NNDEPLOY_INSTALL_INCLUDE_PATH}/nndeploy)
else()
  install(TARGETS ${PLUGIN_BINARY} ${NNDEPLOY_INSTALL_TYPE} DESTINATION ${NNDEPLOY_INSTALL_LIB_PATH})
  install(DIRECTORY ${PLUGIN_ROOT_PATH}/include/nndeploy/keypoint DESTINATION ${NNDEPLOY_INSTALL_INCLUDE_PATH}/nndeploy)
endif()

# appedn list
set(NNDEPLOY_PLUGIN_LIST ${NNDEPLOY_PLUGIN_LIST} ${PLUGIN_BINARY})

# unset
unset(PLUGIN_SOURCE)
unset(PLUGIN_OBJECT)
unset(PLUGIN_BINARY)
