message(STATUS "plugin/detect")

# set
set(PLUGIN_SOURCE)
set(PLUGIN_OBJECT)
set(PLUGIN_BINARY nndeploy_plugin_detect)

# SOURCE
file(GLOB PLUGIN_SOURCE
  "${PLUGIN_ROOT_PATH}/include/nndeploy/detect/*.h"
  "${PLUGIN_ROOT_PATH}/source/nndeploy/detect/*.cc"
)

if(ENABLE_NNDEPLOY_PLUGIN_DETECT_DETR)
  file(GLOB_RECURSE DETR_SOURCE
    "${PLUGIN_ROOT_PATH}/include/nndeploy/detect/detr/*.h"
    "${PLUGIN_ROOT_PATH}/source/nndeploy/detect/detr/*.cc"
  )
  set(PLUGIN_SOURCE ${PLUGIN_SOURCE} ${DETR_SOURCE})
  message(STATUS "  + DETR detect backend")
else()
  message(STATUS "  - DETR detect backend (disabled)")
endif()

if(ENABLE_NNDEPLOY_PLUGIN_DETECT_YOLO)
  file(GLOB_RECURSE YOLO_SOURCE
    "${PLUGIN_ROOT_PATH}/include/nndeploy/detect/yolo/*.h"
    "${PLUGIN_ROOT_PATH}/source/nndeploy/detect/yolo/*.cc"
  )
  set(PLUGIN_SOURCE ${PLUGIN_SOURCE} ${YOLO_SOURCE})
  message(STATUS "  + YOLO detect backend")
else()
  message(STATUS "  - YOLO detect backend (disabled)")
endif()

if(ENABLE_NNDEPLOY_PLUGIN_DETECT_YOLO_OBB)
  file(GLOB_RECURSE OBB_SOURCE
    "${PLUGIN_ROOT_PATH}/include/nndeploy/detect/yolo_obb/*.h"
    "${PLUGIN_ROOT_PATH}/source/nndeploy/detect/yolo_obb/*.cc"
  )
  set(PLUGIN_SOURCE ${PLUGIN_SOURCE} ${OBB_SOURCE})
  message(STATUS "  + YOLO-OBB detect backend")
else()
  message(STATUS "  - YOLO-OBB detect backend (disabled)")
endif()

# YOLO-NAS is now part of the yolo/ directory (included by YOLO_SOURCE above)

if(ENABLE_NNDEPLOY_PLUGIN_DETECT_RF_DETR)
  file(GLOB_RECURSE RF_DETR_SOURCE
    "${PLUGIN_ROOT_PATH}/include/nndeploy/detect/rf_detr/*.h"
    "${PLUGIN_ROOT_PATH}/source/nndeploy/detect/rf_detr/*.cc"
  )
  set(PLUGIN_SOURCE ${PLUGIN_SOURCE} ${RF_DETR_SOURCE})
  message(STATUS "  + RF-DETR detect backend")
else()
  message(STATUS "  - RF-DETR detect backend (disabled)")
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
  nndeploy_install_target(${PLUGIN_BINARY})
  install(DIRECTORY ${PLUGIN_ROOT_PATH}/include/nndeploy/detect DESTINATION ${NNDEPLOY_INSTALL_INCLUDE_PATH}/nndeploy)
else()
  install(TARGETS ${PLUGIN_BINARY} ${NNDEPLOY_INSTALL_TYPE} DESTINATION ${NNDEPLOY_INSTALL_LIB_PATH})
  install(DIRECTORY ${PLUGIN_ROOT_PATH}/include/nndeploy/detect DESTINATION ${NNDEPLOY_INSTALL_INCLUDE_PATH}/nndeploy)
endif()

# appedn list
set(NNDEPLOY_PLUGIN_LIST ${NNDEPLOY_PLUGIN_LIST} ${PLUGIN_BINARY})

# unset
unset(PLUGIN_SOURCE)
unset(PLUGIN_OBJECT)
unset(PLUGIN_BINARY)
