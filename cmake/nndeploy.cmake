
set(NNDEPLOY_THIRD_PARTY_LIBRARY_PATH_SUFFIX lib)

if(SYSTEM.Android)
  list(APPEND NNDEPLOY_SYSTEM_LIBRARY log)
  set(NNDEPLOY_THIRD_PARTY_LIBRARY_PATH_SUFFIX ${ANDROID_ABI})
elseif(SYSTEM.Linux)
elseif(SYSTEM.Darwin)
elseif(SYSTEM.iOS)
elseif(SYSTEM.Windows)
endif()

# ################### common ####################
# # OpenCV
include("${ROOT_PATH}/cmake/opencv.cmake")
# # rapidjson
include("${ROOT_PATH}/cmake/rapidjson.cmake")
# # safetensos_cpp
include("${ROOT_PATH}/cmake/safetensos_cpp.cmake")

# ################### common ####################

# ################### base ####################
# ################### base ####################

# ################### thread ####################
# ################### thread ####################

# ################### cryption ####################
# ################### cryption ####################

# ################### device ####################
# # CUDA & CUDNN
include("${ROOT_PATH}/cmake/cuda.cmake")

# # ascend_cl
include("${ROOT_PATH}/cmake/ascend_cl.cmake")

# # opencl
include("${ROOT_PATH}/cmake/opencl.cmake")

# ################### device ####################

# ################### op ####################
include("${ROOT_PATH}/cmake/onednn.cmake")
include("${ROOT_PATH}/cmake/xnnpack.cmake")
include("${ROOT_PATH}/cmake/qnnpack.cmake")
# ################### op ####################

# ################### net ####################
include("${ROOT_PATH}/cmake/protobuf.cmake")
include("${ROOT_PATH}/cmake/onnx.cmake")

# ################### net ####################

# ################### inference ####################
# # MNN
include("${ROOT_PATH}/cmake/mnn.cmake")

# # tensorrt
include("${ROOT_PATH}/cmake/tensorrt.cmake")

# # onnxruntime
include("${ROOT_PATH}/cmake/onnxruntime.cmake")

# # tnn
include("${ROOT_PATH}/cmake/tnn.cmake")

# # openvino
include("${ROOT_PATH}/cmake/openvino.cmake")

# # ncnn
include("${ROOT_PATH}/cmake/ncnn.cmake")

# # coreml
include("${ROOT_PATH}/cmake/coreml.cmake")

# # paddle-lite
include("${ROOT_PATH}/cmake/paddlelite.cmake")

# # rknn
include("${ROOT_PATH}/cmake/rknn.cmake")

# # snpe
include("${ROOT_PATH}/cmake/snpe.cmake")

# # tvm
include("${ROOT_PATH}/cmake/tvm.cmake")

# ################### inference ####################

# ################### dag ####################
# ################### dag ####################

# ################### plugin ####################
include("${ROOT_PATH}/cmake/tokenizer_cpp.cmake")
# ################### plugin ####################

# ################### test ####################
if(ENABLE_NNDEPLOY_TEST STREQUAL "ON")
  set(NNDEPLOY_TEST_THIRD_PARTY_LIBRARY ${NNDEPLOY_TEST_THIRD_PARTY_LIBRARY} gflags)
endif()

# ################### test ####################

# ################### demo ####################
if(ENABLE_NNDEPLOY_DEMO STREQUAL "ON")
  set(NNDEPLOY_DEMO_THIRD_PARTY_LIBRARY ${NNDEPLOY_DEMO_THIRD_PARTY_LIBRARY} gflags)
endif()

# ################### demo ####################