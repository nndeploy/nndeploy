message(STATUS "model/detect")

# set
set(SOURCE)

# SOURCE
file(GLOB SOURCE
  "${ROOT_PATH}/include/nndeploy/model/detect/*.h"
  "${ROOT_PATH}/source/nndeploy/model/detect/*.cc"
)

if (ENABLE_NNDEPLOY_OPENCV)
  if (ENABLE_NNDEPLOY_MODEL_DETECT_DETR)
    file(GLOB_RECURSE DETR_SOURCE
      "${ROOT_PATH}/include/nndeploy/model/detect/detr/*.h"
      "${ROOT_PATH}/source/nndeploy/model/detect/detr/*.cc"
    )
    set(SOURCE ${SOURCE} ${DETR_SOURCE})
  endif()
  if (ENABLE_NNDEPLOY_MODEL_DEDECT_YOLOV3)
    file(GLOB_RECURSE YOLOV3_SOURCE
      "${ROOT_PATH}/include/nndeploy/model/detect/yolov3/*.h"
      "${ROOT_PATH}/source/nndeploy/model/detect/yolov3/*.cc"
    )
    set(SOURCE ${SOURCE} ${YOLOV3_SOURCE})
  endif()
  if (ENABLE_NNDEPLOY_MODEL_DETECT_YOLO)
    file(GLOB_RECURSE YOLO_SOURCE
      "${ROOT_PATH}/include/nndeploy/model/detect/yolo/*.h"
      "${ROOT_PATH}/source/nndeploy/model/detect/yolo/*.cc"
    )
    set(SOURCE ${SOURCE} ${YOLO_SOURCE})
  endif()
endif()

set(MODEL_SOURCE ${MODEL_SOURCE} ${SOURCE})

# unset
unset(SOURCE)


