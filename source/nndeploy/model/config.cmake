
file(GLOB MODEL_SOURCE
  "${ROOT_PATH}/include/nndeploy/model/*.h"
  "${ROOT_PATH}/source/nndeploy/model/*.cc"
)

if(ENABLE_NNDEPLOY_OPENCV)
  file(GLOB_RECURSE MODEL_PREPROCESS_SOURCE
    "${ROOT_PATH}/include/nndeploy/model/preprocess/*.h"
    "${ROOT_PATH}/source/nndeploy/model/preprocess/*.cc"
  )
  set(MODEL_SOURCE ${MODEL_SOURCE} ${MODEL_PREPROCESS_SOURCE})
endif()

if (ENABLE_NNDEPLOY_MODEL_DETECT)
  include(${ROOT_PATH}/source/nndeploy/model/detect/config.cmake)
endif()

if (ENABLE_NNDEPLOY_MODEL_SEGMENT)
  include(${ROOT_PATH}/source/nndeploy/model/segment/config.cmake)
endif()