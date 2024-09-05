message(STATUS "model/tokenizer")

# set
set(SOURCE)

# SOURCE
file(GLOB SOURCE
  "${ROOT_PATH}/include/nndeploy/model/tokenizer/*.h"
  "${ROOT_PATH}/source/nndeploy/model/tokenizer/*.cc"
)

if (ENABLE_NNDEPLOY_MODEL_TOKENIZER_CPP)
  file(GLOB_RECURSE TOKENIZER_CPP_SOURCE
    "${ROOT_PATH}/include/nndeploy/model/tokenizer/tokenizer_cpp/*.h"
    "${ROOT_PATH}/source/nndeploy/model/tokenizer/tokenizer_cpp/*.cc"
  )
  set(SOURCE ${SOURCE} ${TOKENIZER_CPP_SOURCE})
endif()

set(MODEL_SOURCE ${MODEL_SOURCE} ${SOURCE})

# unset
unset(SOURCE)


