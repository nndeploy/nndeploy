
# plugin cmake config
include_directories(${ROOT_PATH}/plugin/include)
include_directories(${ROOT_PATH}/plugin/source)

# make
set(NNDEPLOY_PLUGIN_DIRECTORY nndeploy_plugin)
set(NNDEPLOY_PLUGIN_LIST)

# plugin path
set(PLUGIN_ROOT_PATH ${ROOT_PATH}/plugin)

# plugin includes
# # preprocess
if(ENABLE_NNDEPLOY_OPENCV AND ENABLE_NNDEPLOY_PLUGIN_PREPROCESS)
  include(${PLUGIN_ROOT_PATH}/source/nndeploy/preprocess/config.cmake)
endif()

# # infer
if(ENABLE_NNDEPLOY_PLUGIN_INFER)
  include(${PLUGIN_ROOT_PATH}/source/nndeploy/infer/config.cmake)
endif()

# # codec
if(ENABLE_NNDEPLOY_OPENCV AND ENABLE_NNDEPLOY_PLUGIN_CODEC)
  include(${PLUGIN_ROOT_PATH}/source/nndeploy/codec/config.cmake)
endif()

# # tokenizer
if(ENABLE_NNDEPLOY_PLUGIN_TOKENIZER)
  include(${PLUGIN_ROOT_PATH}/source/nndeploy/tokenizer/config.cmake)
endif()

# # classification
if(ENABLE_NNDEPLOY_OPENCV AND ENABLE_NNDEPLOY_PLUGIN_CLASSIFICATION)
  include(${PLUGIN_ROOT_PATH}/source/nndeploy/classification/config.cmake)
endif()

# # llm 
if(ENABLE_NNDEPLOY_PLUGIN_LLM)
  include(${PLUGIN_ROOT_PATH}/source/nndeploy/llm/config.cmake)
endif()

# # detect
if(ENABLE_NNDEPLOY_OPENCV AND ENABLE_NNDEPLOY_PLUGIN_DETECT)
  include(${PLUGIN_ROOT_PATH}/source/nndeploy/detect/config.cmake)
endif()

# # segment
if(ENABLE_NNDEPLOY_OPENCV AND ENABLE_NNDEPLOY_PLUGIN_SEGMENT)
  include(${PLUGIN_ROOT_PATH}/source/nndeploy/segment/config.cmake)
endif()

# # stable_diffusion
if(ENABLE_NNDEPLOY_OPENCV AND ENABLE_NNDEPLOY_PLUGIN_STABLE_DIFFUSION)
  include(${PLUGIN_ROOT_PATH}/source/nndeploy/stable_diffusion/config.cmake)
endif()

# # super_resolution
if(ENABLE_NNDEPLOY_OPENCV AND ENABLE_NNDEPLOY_PLUGIN_SUPER_RESOLUTION)
  include(${PLUGIN_ROOT_PATH}/source/nndeploy/super_resolution/config.cmake)
endif()

# # track
if(ENABLE_NNDEPLOY_OPENCV AND ENABLE_NNDEPLOY_PLUGIN_TRACK)
  include(${PLUGIN_ROOT_PATH}/source/nndeploy/track/config.cmake)
endif()

# appedn list
message(STATUS "NNDEPLOY_PLUGIN_LIST: ${NNDEPLOY_PLUGIN_LIST}")
