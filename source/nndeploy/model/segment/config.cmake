message(STATUS "model/segment")

# set
set(SOURCE)

# SOURCE
file(GLOB SOURCE
  "${ROOT_PATH}/include/nndeploy/model/segment/*.h"
  "${ROOT_PATH}/source/nndeploy/model/segment/*.cc"
)

if (ENABLE_NNDEPLOY_OPENCV)
  if (ENABLE_NNDEPLOY_MODEL_SEGMENT_SEGMENT_ANYTHING)
    file(GLOB_RECURSE SEGMENT_ANYTHING_SOURCE
      "${ROOT_PATH}/include/nndeploy/model/segment/segment_anything/*.h"
      "${ROOT_PATH}/source/nndeploy/model/segment/segment_anything/*.cc"
    )
    set(SOURCE ${SOURCE} ${SEGMENT_ANYTHING_SOURCE})
  endif()
endif()

set(MODEL_SOURCE ${MODEL_SOURCE} ${SOURCE})

# unset
unset(SOURCE)


