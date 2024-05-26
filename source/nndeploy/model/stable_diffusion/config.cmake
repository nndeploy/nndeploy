message(STATUS "model/stable_diffusion")

# set
set(SOURCE)

# SOURCE
file(GLOB SOURCE
  "${ROOT_PATH}/include/nndeploy/model/stable_diffusion/*.h"
  "${ROOT_PATH}/source/nndeploy/model/stable_diffusion/*.cc"
)

set(MODEL_SOURCE ${MODEL_SOURCE} ${SOURCE})

# unset
unset(SOURCE)