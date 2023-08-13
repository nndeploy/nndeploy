message(STATUS "model/detect")

# set
set(SOURCE)

# SOURCE
file(GLOB SOURCE
  "${ROOT_PATH}/include/nndeploy/model/detect/*.h"
  "${ROOT_PATH}/source/nndeploy/model/detect/*.cc"
)

if (ENABLE_NNDEPLOY_OPENCV)
  message(STATUS "model/detect: opencv")
  if (ENABLE_NNDEPLOY_MODEL_DETECT_DETR)
    file(GLOB_RECURSE DETR_SOURCE
      "${ROOT_PATH}/include/nndeploy/model/detect/detr/*.h"
      "${ROOT_PATH}/source/nndeploy/model/detect/detr/*.cc"
    )
    set(SOURCE ${SOURCE} ${DETR_SOURCE})
  endif()
  if (ENABLE_NNDEPLOY_MODEL_DEDECT_YOLOV5)
    file(GLOB_RECURSE YOLOV5_SOURCE
      "${ROOT_PATH}/include/nndeploy/model/detect/yolov5/*.h"
      "${ROOT_PATH}/source/nndeploy/model/detect/yolov5/*.cc"
    )
    set(SOURCE ${SOURCE} ${YOLOV5_SOURCE})
  endif()
  if (ENABLE_NNDEPLOY_MODEL_DETECT_MEITUAN_YOLOV6)
    file(GLOB_RECURSE MEITUAN_YOLOV6_SOURCE
      "${ROOT_PATH}/include/nndeploy/model/detect/meituan_yolov6/*.h"
      "${ROOT_PATH}/source/nndeploy/model/detect/meituan_yolov6/*.cc"
    )
    set(SOURCE ${SOURCE} ${MEITUAN_YOLOV6_SOURCE})
  endif()
endif()

set(MODEL_SOURCE ${MODEL_SOURCE} ${SOURCE})

# unset
unset(SOURCE)


