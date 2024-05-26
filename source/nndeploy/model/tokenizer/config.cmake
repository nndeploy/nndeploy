message(STATUS "model/tokenizer")

# set
set(SOURCE)

# SOURCE
file(GLOB SOURCE
  "${ROOT_PATH}/include/nndeploy/model/tokenizer/*.h"
  "${ROOT_PATH}/source/nndeploy/model/tokenizer/*.cc"
)

set(MODEL_SOURCE ${MODEL_SOURCE} ${SOURCE})

# unset
unset(SOURCE)


