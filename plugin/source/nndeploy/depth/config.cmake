message(STATUS "plugin/depth")

# set
set(PLUGIN_SOURCE)
set(PLUGIN_OBJECT)
set(PLUGIN_BINARY nndeploy_plugin_depth)

# 深度插件源码（通用层：DrawDepth 等公共节点位于 depth/ 根目录）
file(GLOB PLUGIN_SOURCE
  "${PLUGIN_ROOT_PATH}/include/nndeploy/depth/*.h"
  "${PLUGIN_ROOT_PATH}/source/nndeploy/depth/*.cc"
)

# 子开关：DepthAnything 后端

if(ENABLE_NNDEPLOY_PLUGIN_DEPTH_DEPTH_ANYTHING)
  file(GLOB DEPTH_ANYTHING_SOURCE
    "${PLUGIN_ROOT_PATH}/include/nndeploy/depth/depth_anything/*.h"
    "${PLUGIN_ROOT_PATH}/source/nndeploy/depth/depth_anything/*.cc"
  )
  set(PLUGIN_SOURCE ${PLUGIN_SOURCE} ${DEPTH_ANYTHING_SOURCE})
  message(STATUS "  + Depth Anything backend")
else()
  message(STATUS "  - Depth Anything backend (disabled)")
endif()

# 子开关：YOLO Depth 后端（YOLO26n/s/m/l/x-depth 深度估计）
if(ENABLE_NNDEPLOY_PLUGIN_DEPTH_YOLO_DEPTH)
  file(GLOB YOLO_DEPTH_SOURCE
    "${PLUGIN_ROOT_PATH}/include/nndeploy/depth/yolo_depth/*.h"
    "${PLUGIN_ROOT_PATH}/source/nndeploy/depth/yolo_depth/*.cc"
  )
  set(PLUGIN_SOURCE ${PLUGIN_SOURCE} ${YOLO_DEPTH_SOURCE})
  message(STATUS "  + YOLO Depth backend")
else()
  message(STATUS "  - YOLO Depth backend (disabled)")
endif()

if(PLUGIN_SOURCE)
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
    install(DIRECTORY ${PLUGIN_ROOT_PATH}/include/nndeploy/depth DESTINATION ${NNDEPLOY_INSTALL_INCLUDE_PATH}/nndeploy)
  else()
    install(TARGETS ${PLUGIN_BINARY} ${NNDEPLOY_INSTALL_TYPE} DESTINATION ${NNDEPLOY_INSTALL_LIB_PATH})
    install(DIRECTORY ${PLUGIN_ROOT_PATH}/include/nndeploy/depth DESTINATION ${NNDEPLOY_INSTALL_INCLUDE_PATH}/nndeploy)
  endif()

  # appedn list
  set(NNDEPLOY_PLUGIN_LIST ${NNDEPLOY_PLUGIN_LIST} ${PLUGIN_BINARY})
else()
  message(STATUS "  - depth plugin (no backend enabled)")
endif()

