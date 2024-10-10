message(STATUS "model/detect")

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
endif()

if(ENABLE_NNDEPLOY_PLUGIN_DETECT_YOLO)
  file(GLOB_RECURSE YOLO_SOURCE
    "${PLUGIN_ROOT_PATH}/include/nndeploy/detect/yolo/*.h"
    "${PLUGIN_ROOT_PATH}/source/nndeploy/detect/yolo/*.cc"
  )
  set(PLUGIN_SOURCE ${PLUGIN_SOURCE} ${YOLO_SOURCE})
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

# # NNDEPLOY_PLUGIN_THIRD_PARTY_LIBRARY
target_link_libraries(${PLUGIN_BINARY} ${NNDEPLOY_PLUGIN_THIRD_PARTY_LIBRARY})

# # install
if(SYSTEM.Windows)
  install(TARGETS ${PLUGIN_BINARY} ${NNDEPLOY_INSTALL_TYPE} DESTINATION ${NNDEPLOY_INSTALL_PATH})
else()
  install(TARGETS ${PLUGIN_BINARY} ${NNDEPLOY_INSTALL_TYPE} DESTINATION ${NNDEPLOY_INSTALL_LIB_PATH})
endif()

# appedn list
set(NNDEPLOY_PLUGIN_LIST ${NNDEPLOY_PLUGIN_LIST} ${PLUGIN_BINARY})

# unset
unset(PLUGIN_SOURCE)
unset(PLUGIN_OBJECT)
unset(PLUGIN_BINARY)
