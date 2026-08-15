
message(STATUS "plugin/segment")

# set
set(PLUGIN_SOURCE)
set(PLUGIN_OBJECT)
set(PLUGIN_BINARY nndeploy_plugin_segment)

# SOURCE
file(GLOB PLUGIN_SOURCE
  "${PLUGIN_ROOT_PATH}/include/nndeploy/segment/*.h"
  "${PLUGIN_ROOT_PATH}/source/nndeploy/segment/*.cc"
)

if(ENABLE_NNDEPLOY_PLUGIN_SEGMENT_SEGMENT_ANYTHING)
  file(GLOB_RECURSE SEGMENT_ANYTHING_SOURCE
    "${PLUGIN_ROOT_PATH}/include/nndeploy/segment/segment_anything/*.h"
    "${PLUGIN_ROOT_PATH}/source/nndeploy/segment/segment_anything/*.cc"
  )
  set(PLUGIN_SOURCE ${PLUGIN_SOURCE} ${SEGMENT_ANYTHING_SOURCE})
  message(STATUS "  + SEGMENT_ANYTHING segment backend")
else()
  message(STATUS "  - SEGMENT_ANYTHING segment backend (disabled)")
endif()

if(ENABLE_NNDEPLOY_PLUGIN_SEGMENT_RMBG)
  file(GLOB_RECURSE RMBG_SOURCE
    "${PLUGIN_ROOT_PATH}/include/nndeploy/segment/rmbg/*.h"
    "${PLUGIN_ROOT_PATH}/source/nndeploy/segment/rmbg/*.cc"
  )
  set(PLUGIN_SOURCE ${PLUGIN_SOURCE} ${RMBG_SOURCE})
  message(STATUS "  + RMBG segment backend")
else()
  message(STATUS "  - RMBG segment backend (disabled)")
endif()

if(ENABLE_NNDEPLOY_PLUGIN_SEGMENT_YOLO_SEG)
  file(GLOB_RECURSE YOLO_SEG_SOURCE
    "${PLUGIN_ROOT_PATH}/include/nndeploy/segment/yolo_seg/*.h"
    "${PLUGIN_ROOT_PATH}/source/nndeploy/segment/yolo_seg/*.cc"
  )
  set(PLUGIN_SOURCE ${PLUGIN_SOURCE} ${YOLO_SEG_SOURCE})
  message(STATUS "  + YOLO-SEG segment backend")
else()
  message(STATUS "  - YOLO-SEG segment backend (disabled)")
endif()

if(ENABLE_NNDEPLOY_PLUGIN_SEGMENT_RF_DETR_SEG)
  file(GLOB_RECURSE RF_DETR_SEG_SOURCE
    "${PLUGIN_ROOT_PATH}/include/nndeploy/segment/rf_detr_seg/*.h"
    "${PLUGIN_ROOT_PATH}/source/nndeploy/segment/rf_detr_seg/*.cc"
  )
  set(PLUGIN_SOURCE ${PLUGIN_SOURCE} ${RF_DETR_SEG_SOURCE})
  message(STATUS "  + RF-DETR-SEG segment backend")
else()
  message(STATUS "  - RF-DETR-SEG segment backend (disabled)")
endif()

if(ENABLE_NNDEPLOY_PLUGIN_SEGMENT_SAM2)
  file(GLOB SAM2_SOURCE
    "${PLUGIN_ROOT_PATH}/include/nndeploy/segment/segment_anything/sam2.h"
    "${PLUGIN_ROOT_PATH}/source/nndeploy/segment/segment_anything/sam2.cc"
  )
  set(PLUGIN_SOURCE ${PLUGIN_SOURCE} ${SAM2_SOURCE})
  message(STATUS "  + SAM2 segment backend")
else()
  message(STATUS "  - SAM2 segment backend (disabled)")
endif()

if(ENABLE_NNDEPLOY_PLUGIN_SEGMENT_SAM3)
  file(GLOB SAM3_SOURCE
    "${PLUGIN_ROOT_PATH}/include/nndeploy/segment/segment_anything/sam3.h"
    "${PLUGIN_ROOT_PATH}/source/nndeploy/segment/segment_anything/sam3.cc"
  )
  set(PLUGIN_SOURCE ${PLUGIN_SOURCE} ${SAM3_SOURCE})
  message(STATUS "  + SAM3 segment backend")
else()
  message(STATUS "  - SAM3 segment backend (disabled)")
endif()

# # TARGET
add_library(${PLUGIN_BINARY} ${NNDEPLOY_LIB_TYPE} ${PLUGIN_SOURCE} ${PLUGIN_OBJECT})

# # DIRECTORY
set_property(TARGET ${PLUGIN_BINARY} PROPERTY FOLDER ${NNDEPLOY_PLUGIN_DIRECTORY})

# # DEPEND_LIBRARY
target_link_libraries(${PLUGIN_BINARY} ${NNDEPLOY_DEPEND_LIBRARY})

# # SYSTEM_LIBRARY
target_link_libraries(${PLUGIN_BINARY} ${NNDEPLOY_SYSTEM_LIBRARY})

# # THIRD_PARTY_LIBRARY
target_link_libraries(${PLUGIN_BINARY} ${NNDEPLOY_THIRD_PARTY_LIBRARY})

# # NNDEPLOY_FRAMEWORK_BINARY
target_link_libraries(${PLUGIN_BINARY} ${NNDEPLOY_FRAMEWORK_BINARY})
target_link_libraries(${PLUGIN_BINARY} nndeploy_plugin_preprocess)
target_link_libraries(${PLUGIN_BINARY} nndeploy_plugin_infer)
target_link_libraries(${PLUGIN_BINARY} nndeploy_plugin_detect)

# # NNDEPLOY_PLUGIN_THIRD_PARTY_LIBRARY
target_link_libraries(${PLUGIN_BINARY} ${NNDEPLOY_PLUGIN_THIRD_PARTY_LIBRARY})

# # install
if(SYSTEM_Windows)
  nndeploy_install_target(${PLUGIN_BINARY})
  install(DIRECTORY ${PLUGIN_ROOT_PATH}/include/nndeploy/segment DESTINATION ${NNDEPLOY_INSTALL_INCLUDE_PATH}/nndeploy)
else()
  install(TARGETS ${PLUGIN_BINARY} ${NNDEPLOY_INSTALL_TYPE} DESTINATION ${NNDEPLOY_INSTALL_LIB_PATH})
  install(DIRECTORY ${PLUGIN_ROOT_PATH}/include/nndeploy/segment DESTINATION ${NNDEPLOY_INSTALL_INCLUDE_PATH}/nndeploy)
endif()

# appedn list
set(NNDEPLOY_PLUGIN_LIST ${NNDEPLOY_PLUGIN_LIST} ${PLUGIN_BINARY})

# unset
unset(PLUGIN_SOURCE)
unset(PLUGIN_OBJECT)
unset(PLUGIN_BINARY)
