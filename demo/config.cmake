message(STATUS "Building nndeploy demo")

# framework
if(ENABLE_NNDEPLOY_BASE)
  include(${ROOT_PATH}/demo/base/config.cmake)
endif()

if(ENABLE_NNDEPLOY_THREAD_POOL)
  include(${ROOT_PATH}/demo/thread_pool/config.cmake)
endif()

if(ENABLE_NNDEPLOY_DEVICE)
  include(${ROOT_PATH}/demo/device/config.cmake)

  if(ENABLE_NNDEPLOY_DEVICE_ASCEND_CL)
    include(${ROOT_PATH}/demo/ascend_cl/config.cmake)
  endif()

  if(ENABLE_NNDEPLOY_OP_ASCEND_C)
    include(${ROOT_PATH}/demo/ascendc/config.cmake)
  endif()
endif()

if(ENABLE_NNDEPLOY_IR)
  include(${ROOT_PATH}/demo/ir/config.cmake)
endif()

if(ENABLE_NNDEPLOY_IR)
  include(${ROOT_PATH}/demo/interpret/config.cmake)
endif()

if(ENABLE_NNDEPLOY_OP)
  include(${ROOT_PATH}/demo/op/config.cmake)
endif()

if(ENABLE_NNDEPLOY_OP_ASCEND_C)
  include(${ROOT_PATH}/demo/ascendc_dev/config.cmake)
endif()

if(ENABLE_NNDEPLOY_NET)
  include(${ROOT_PATH}/demo/net/config.cmake)
  include(${ROOT_PATH}/demo/optimizer/config.cmake)
  include(${ROOT_PATH}/demo/tensor_pool/config.cmake)
  include(${ROOT_PATH}/demo/tensor_pool_multi_net/config.cmake)
endif()

if(ENABLE_NNDEPLOY_INFERENCE)
  include(${ROOT_PATH}/demo/inference/config.cmake)
endif()

if(ENABLE_NNDEPLOY_DAG)
  include(${ROOT_PATH}/demo/dag/config.cmake)
  include(${ROOT_PATH}/demo/run_json/config.cmake)
endif()

# plugin
if(ENABLE_NNDEPLOY_OPENCV AND ENABLE_NNDEPLOY_PLUGIN_PREPROCESS)
  include(${ROOT_PATH}/demo/preprocess/config.cmake)
endif()

if(ENABLE_NNDEPLOY_PLUGIN_INFER)
  include(${ROOT_PATH}/demo/infer/config.cmake)
endif()

if(ENABLE_NNDEPLOY_OPENCV AND ENABLE_NNDEPLOY_PLUGIN_CODEC)
  include(${ROOT_PATH}/demo/codec/config.cmake)
endif()

if(ENABLE_NNDEPLOY_OPENCV AND ENABLE_NNDEPLOY_PLUGIN_CLASSIFICATION)
  include(${ROOT_PATH}/demo/classification/config.cmake)
endif()

if(ENABLE_NNDEPLOY_OPENCV AND ENABLE_NNDEPLOY_PLUGIN_DETECT)
  include(${ROOT_PATH}/demo/detect/config.cmake)
endif()

if(ENABLE_NNDEPLOY_OPENCV AND ENABLE_NNDEPLOY_PLUGIN_SEGMENT)
  include(${ROOT_PATH}/demo/segment/config.cmake)
endif()

if(ENABLE_NNDEPLOY_OPENCV AND ENABLE_NNDEPLOY_PLUGIN_SEGMENT_SEGMENT_ANYTHING)
  include(${ROOT_PATH}/demo/segment_anything/config.cmake)
endif()

if(ENABLE_NNDEPLOY_PLUGIN_TOKENIZER_CPP)
  include(${ROOT_PATH}/demo/tokenizer_cpp/config.cmake)
endif()

if(ENABLE_NNDEPLOY_OPENCV AND ENABLE_NNDEPLOY_PLUGIN_STABLE_DIFFUSION)
  include(${ROOT_PATH}/demo/stable_diffusion/config.cmake)
endif()

if(ENABLE_NNDEPLOY_PLUGIN_LLM)
  include(${ROOT_PATH}/demo/llama/config.cmake)
  include(${ROOT_PATH}/demo/qwen/config.cmake)
endif()

nndeploy_option(ENABLE_NNDEPLOY_DEMO_RESNET "ENABLE_NNDEPLOY_DEMO_RESNET" OFF)
if(ENABLE_NNDEPLOY_DEMO_RESNET)
  include(${ROOT_PATH}/demo/resnet/config.cmake)
endif()

if(ENABLE_NNDEPLOY_PLUGIN_SUPER_RESOLUTION)
  include(${ROOT_PATH}/demo/super_resolution/config.cmake)
endif()

if(ENABLE_NNDEPLOY_PLUGIN_OCR)
  include(${ROOT_PATH}/demo/ocr/config.cmake)
endif()

if(ENABLE_NNDEPLOY_PLUGIN_TRACK)
  include(${ROOT_PATH}/demo/track/config.cmake)
endif()

if(ENABLE_NNDEPLOY_PLUGIN_MATTING)
  include(${ROOT_PATH}/demo/matting/config.cmake)
endif()

