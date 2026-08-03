if(ENABLE_NNDEPLOY_PLUGIN_DETECT_RF_DETR)
  file(GLOB_RECURSE RF_DETR_SOURCE
    "${PLUGIN_ROOT_PATH}/include/nndeploy/detect/rf_detr/*.h"
    "${PLUGIN_ROOT_PATH}/source/nndeploy/detect/rf_detr/*.cc"
  )
  set(PLUGIN_SOURCE ${PLUGIN_SOURCE} ${RF_DETR_SOURCE})
endif()
