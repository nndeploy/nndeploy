# --------------------------------------------------------------------
# Template custom cmake config for compiling
#
# This file is used to override the build sets in build.
# If you want to change the config, please use the following
# steps. Assume you are off the root directory. First copy the this
# file so that any local changes will be ignored by git
#
# $ mkdir build
# $ cp cmake/config.cmake build
# $ cd build
# $ vim config.cmake
# $ cmake ..
# $ make -j
# --------------------------------------------------------------------

# IR ONNX
set(ENABLE_NNDEPLOY_IR_ONNX OFF) # Support generating IR directly from ONNX models, disabled by default

# Device Backend Options (Enable as Needed, All Disabled by Default, No Device Backend Dependencies)
set(ENABLE_NNDEPLOY_DEVICE_CUDA OFF) # Whether to enable device CUDA, default is OFF
set(ENABLE_NNDEPLOY_DEVICE_ROCM OFF) # Whether to enable device ROCM, default is OFF
set(ENABLE_NNDEPLOY_DEVICE_SYCL OFF) # Whether to enable device SYCL, default is OFF
set(ENABLE_NNDEPLOY_DEVICE_OPENCL OFF) # Whether to enable device OpenCL, default is OFF
set(ENABLE_NNDEPLOY_DEVICE_OPENGL OFF) # Whether to enable device OpenGL, default is OFF
set(ENABLE_NNDEPLOY_DEVICE_METAL OFF) # Whether to enable device Metal, default is OFF
set(ENABLE_NNDEPLOY_DEVICE_VULKAN OFF) # Whether to enable device Vulkan, default is OFF
set(ENABLE_NNDEPLOY_DEVICE_HEXAGON OFF) # Whether to enable device Hexagon, default is OFF
set(ENABLE_NNDEPLOY_DEVICE_MTK_VPU OFF) # Whether to enable device MTK VPU, default is OFF
set(ENABLE_NNDEPLOY_DEVICE_ASCEND_CL OFF) # Whether to enable device Ascend CL, default is OFF
set(ENABLE_NNDEPLOY_DEVICE_APPLE_NPU OFF) # Whether to enable device Apple NPU, default is OFF
set(ENABLE_NNDEPLOY_DEVICE_QUALCOMM_NPU OFF) # Whether to enable device Qualcomm NPU, default is OFF
set(ENABLE_NNDEPLOY_DEVICE_MTK_NPU OFF) # Whether to enable device MTK NPU, default is OFF
set(ENABLE_NNDEPLOY_DEVICE_SOPHON_NPU OFF) # Whether to enable device Sophon NPU, default is OFF

# Operator Backend Options (Enable as Needed, All Disabled by Default, No Operator Backend Dependencies)
set(ENABLE_NNDEPLOY_DEVICE_CUDNN OFF) # Whether to enable operator CUDNN, default is OFF
set(ENABLE_NNDEPLOY_DEVICE_X86_ONEDNN OFF) # Whether to enable operator X86_ONEDNN, default is OFF
set(ENABLE_NNDEPLOY_DEVICE_ARM_XNNPACK OFF) # Whether to enable operator ARM_XNNPACK, default is OFF
set(ENABLE_NNDEPLOY_DEVICE_ARM_QNNPACK OFF) # Whether to enable operator ARM_QNNPACK, default is OFF

# Inference Backend Options (Enable as Needed, All Disabled by Default, No Inference Backend Dependencies)
set(ENABLE_NNDEPLOY_INFERENCE_TENSORRT OFF) # Whether to enable INFERENCE TENSORRT, default is OFF
set(ENABLE_NNDEPLOY_INFERENCE_OPENVINO OFF) # Whether to enable INFERENCE OPENVINO, default is OFF
set(ENABLE_NNDEPLOY_INFERENCE_COREML OFF) # Whether to enable INFERENCE COREML, default is OFF
set(ENABLE_NNDEPLOY_INFERENCE_TFLITE OFF) # Whether to enable INFERENCE TFLITE, default is OFF
set(ENABLE_NNDEPLOY_INFERENCE_ONNXRUNTIME "tool/script/third_party/onnxruntime1.18.0")
set(ENABLE_NNDEPLOY_INFERENCE_NCNN OFF) # Whether to enable INFERENCE NCNN, default is OFF
set(ENABLE_NNDEPLOY_INFERENCE_TNN OFF) # Whether to enable INFERENCE TNN, default is OFF
set(ENABLE_NNDEPLOY_INFERENCE_MNN OFF) # Whether to enable INFERENCE MNN, default is OFF
set(ENABLE_NNDEPLOY_INFERENCE_TVM OFF) # Whether to enable INFERENCE TVM, default is OFF
set(ENABLE_NNDEPLOY_INFERENCE_PADDLELITE OFF) # Whether to enable INFERENCE PADDLELITE, default is OFF
set(ENABLE_NNDEPLOY_INFERENCE_RKNN_TOOLKIT_1 OFF) # Whether to enable INFERENCE RKNN_TOOLKIT_1, default is OFF
set(ENABLE_NNDEPLOY_INFERENCE_RKNN_TOOLKIT_2 OFF) # Whether to enable INFERENCE RKNN_TOOLKIT_2, default is OFF
set(ENABLE_NNDEPLOY_INFERENCE_ASCEND_CL OFF) # Whether to enable INFERENCE ASCEND_CL, default is OFF
set(ENABLE_NNDEPLOY_INFERENCE_SNPE OFF) # Whether to enable INFERENCE SNPE, default is OFF
set(ENABLE_NNDEPLOY_INFERENCE_QNN OFF) # Whether to enable INFERENCE QNN, default is OFF
set(ENABLE_NNDEPLOY_INFERENCE_SOPHON OFF) # Whether to enable INFERENCE SOPHON, default is OFF
set(ENABLE_NNDEPLOY_INFERENCE_TORCH OFF) # Whether to enable INFERENCE TORCH, default is OFF
set(ENABLE_NNDEPLOY_INFERENCE_TENSORFLOW OFF) # Whether to enable INFERENCE TENSORFLOW, default is OFF
set(ENABLE_NNDEPLOY_INFERENCE_NEUROPILOT OFF) # Whether to enable INFERENCE NEUROPILOT, default is OFF

# Algorithm Plugin Options (Recommended to use default configuration, traditional CV algorithms enabled, language and text-to-image algorithms disabled by default)
## OpenCV
# set(ENABLE_NNDEPLOY_OPENCV "path/to/opencv") # Link OpenCV by specifying the path
# set(NNDEPLOY_OPENCV_LIBS "opencv_world4100") # Specific OpenCV library names to link, such as opencv_world4100, opencv_java4, etc.
set(ENABLE_NNDEPLOY_OPENCV "tool/script/third_party/opencv4.10.0") # Whether to link the third-party OpenCV library, default is ON
# Includes complete functional modules such as image display, camera calibration, feature detection, and KalmanFilter tracking functionality
set(NNDEPLOY_OPENCV_LIBS opencv_core opencv_imgproc opencv_imgcodecs opencv_videoio opencv_highgui opencv_video opencv_dnn opencv_calib3d opencv_features2d opencv_flann)
set(NNDEPLOY_OPENCV_VERSION 4100)

## Tokenizer-cpp
set(ENABLE_NNDEPLOY_PLUGIN_TOKENIZER_CPP ON) # Whether to enable C++ tokenizer plugin, default is OFF

## Language Model
set(ENABLE_NNDEPLOY_PLUGIN_LLM ON) # Whether to enable language model plugin, default is OFF

## Stable Diffusion
set(ENABLE_NNDEPLOY_PLUGIN_STABLE_DIFFUSION ON) # Whether to enable text-to-image plugin, default is OFF