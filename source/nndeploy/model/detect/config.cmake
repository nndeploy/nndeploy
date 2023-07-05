# set
set(SOURCE)
# SOURCE
file(GLOB SOURCE
  "${ROOT_PATH}/include/nndeploy/model/detect/*.h"
  "${ROOT_PATH}/source/nndeploy/model/detect/*.cc"
  )
if (ENABLE_NNDEPLOY_OPENCV)
  file(GLOB_RECURSE OPENCV_SOURCE
    "${ROOT_PATH}/include/nndeploy/model/detect/opencv/*.h"
    "${ROOT_PATH}/source/nndeploy/model/detect/opencv/*.cc"
  )
  set(SOURCE ${SOURCE} ${OPENCV_SOURCE})
endif()
set(MODEL_SOURCE ${MODEL_SOURCE} ${SOURCE})
# unset
unset(SOURCE)


