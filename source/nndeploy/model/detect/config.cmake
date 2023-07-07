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
  if (ENABLE_NNDEPLOY_MODEL_DETECT_DETR)
    file(GLOB_RECURSE DETR_OPENCV_SOURCE
      "${ROOT_PATH}/include/nndeploy/model/detect/opencv/detr/*.h"
      "${ROOT_PATH}/source/nndeploy/model/detect/opencv/detr/*.cc"
    )
    set(OPENCV_SOURCE ${OPENCV_SOURCE} ${DETR_OPENCV_SOURCE})
  endif()
  if (ENABLE_NNDEPLOY_MODEL_DEDECT_YOLOV5)
    file(GLOB_RECURSE YOLOV5_OPENCV_SOURCE
      "${ROOT_PATH}/include/nndeploy/model/detect/opencv/yolov5/*.h"
      "${ROOT_PATH}/source/nndeploy/model/detect/opencv/yolov5/*.cc"
    )
    set(OPENCV_SOURCE ${OPENCV_SOURCE} ${YOLOV5_OPENCV_SOURCE})
  endif()
  set(SOURCE ${SOURCE} ${OPENCV_SOURCE})
endif()
set(MODEL_SOURCE ${MODEL_SOURCE} ${SOURCE})
# unset
unset(SOURCE)


