message(STATUS "model/stable_diffusion")

# set
set(SOURCE)

# SOURCE
file(GLOB SOURCE
  "${ROOT_PATH}/include/nndeploy/model/stable_diffusion/*.h"
  "${ROOT_PATH}/source/nndeploy/model/stable_diffusion/*.cc"
)

if (ENABLE_NNDEPLOY_OPENCV)
  if (ENABLE_NNDEPLOY_MODEL_STABLE_DIFFUSION_STABLE_DIFFUSION)
    file(GLOB_RECURSE STABLE_DIFFUSION_SOURCE
      "${ROOT_PATH}/include/nndeploy/model/stable_diffusion/stable_diffusion/*.h"
      "${ROOT_PATH}/source/nndeploy/model/stable_diffusion/stable_diffusion/*.cc"
    )
    set(SOURCE ${SOURCE} ${STABLE_DIFFUSION_SOURCE})
  endif()
endif()

set(MODEL_SOURCE ${MODEL_SOURCE} ${SOURCE})

# unset
unset(SOURCE)