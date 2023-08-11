
#################### common ####################
## OpenCV
include("${ROOT_PATH}/cmake/opencv.cmake")
#################### common ####################

#################### base ####################
#################### base ####################

#################### thread ####################
#################### thread ####################

#################### cryption ####################
#################### cryption ####################

#################### device ####################
## CUDA & CUDNN
include("${ROOT_PATH}/cmake/cuda.cmake")
#################### device ####################

#################### op ####################
#################### op ####################

#################### forward ####################
#################### forward ####################

#################### inference ####################
## MNN
include("${ROOT_PATH}/cmake/mnn.cmake")
## tensorrt
include("${ROOT_PATH}/cmake/tensorrt.cmake")
## onnxruntime
include("${ROOT_PATH}/cmake/onnxruntime.cmake")
## tnn
include("${ROOT_PATH}/cmake/tnn.cmake")
## openvino
include("${ROOT_PATH}/cmake/openvino.cmake")
#################### inference ####################

#################### pipeline ####################
#################### pipeline ####################

#################### model ####################
#################### model ####################

message(STATUS "NNDEPLOY_THIRD_PARTY_LIBRARY: ${NNDEPLOY_THIRD_PARTY_LIBRARY}")