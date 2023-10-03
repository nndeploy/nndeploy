
set(NNDEPLOY_THIRD_PARTY_LIBRARY_PATH_SUFFIX lib)

if(SYSTEM.Android)
  list(APPEND NNDEPLOY_SYSTEM_LIBRARY log)
  set(NNDEPLOY_THIRD_PARTY_LIBRARY_PATH_SUFFIX ${ANDROID_ABI})
  message(STATUS "NNDEPLOY_THIRD_PARTY_LIBRARY_PATH_SUFFIX: ${NNDEPLOY_THIRD_PARTY_LIBRARY_PATH_SUFFIX}")
elseif(SYSTEM.Linux)
elseif(SYSTEM.Darwin)
elseif(SYSTEM.iOS)
elseif(SYSTEM.Windows)
endif()

# ################### common ####################
# # OpenCV
include("${ROOT_PATH}/cmake/opencv.cmake")

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

# ################### device ####################

# ################### op ####################
# ################### op ####################

# ################### forward ####################
# ################### forward ####################

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

# ################### inference ####################

# ################### pipeline ####################
# ################### pipeline ####################

# ################### model ####################
# ################### model ####################
message(STATUS "NNDEPLOY_THIRD_PARTY_LIBRARY: ${NNDEPLOY_THIRD_PARTY_LIBRARY}")