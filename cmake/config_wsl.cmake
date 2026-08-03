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
set(ENABLE_NNDEPLOY_DEVICE_AXERA_NPU OFF) # Whether to enable device Axera NPU, default is OFF

# Operator Backend Options (Enable as Needed, All Disabled by Default, No Operator Backend Dependencies)
set(ENABLE_NNDEPLOY_DEVICE_CUDNN OFF) # Whether to enable operator CUDNN, default is OFF
set(ENABLE_NNDEPLOY_DEVICE_X86_ONEDNN OFF) # Whether to enable operator X86_ONEDNN, default is OFF
set(ENABLE_NNDEPLOY_DEVICE_ARM_XNNPACK OFF) # Whether to enable operator ARM_XNNPACK, default is OFF
set(ENABLE_NNDEPLOY_DEVICE_ARM_QNNPACK OFF) # Whether to enable operator ARM_QNNPACK, default is OFF

# Inference Backend Options (Enable as Needed, All Disabled by Default, No Inference Backend Dependencies)
set(ENABLE_NNDEPLOY_INFERENCE_TENSORRT OFF) # Whether to enable INFERENCE TENSORRT, default is OFF
set(ENABLE_NNDEPLOY_INFERENCE_OPENVINO "/mnt/e/Gdsc/projects/dev/devlibs/linux-x64/openvino_toolkit_ubuntu24_2026.2.0.21903.52ddc073857_x86_64")
set(ENABLE_NNDEPLOY_INFERENCE_COREML OFF) # Whether to enable INFERENCE COREML, default is OFF
set(ENABLE_NNDEPLOY_INFERENCE_TFLITE OFF) # Whether to enable INFERENCE TFLITE, default is OFF
set(ENABLE_NNDEPLOY_INFERENCE_ONNXRUNTIME "/mnt/e/Gdsc/projects/dev/devlibs/linux-x64/onnxruntime-linux-x64-1.27.1")
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
set(ENABLE_NNDEPLOY_INFERENCE_AXERA OFF) # Whether to enable INFERENCE AXERA, default is OFF
set(ENABLE_NNDEPLOY_INFERENCE_LYNXI OFF) # Whether to enable INFERENCE LYNXI, default is OFF

# Algorithm Plugin Options (Recommended to use default configuration, traditional CV algorithms enabled, language and text-to-image algorithms disabled by default)
## OpenCV
# set(ENABLE_NNDEPLOY_OPENCV "path/to/opencv") # 通过路径的方式链接OpenCV
# set(NNDEPLOY_OPENCV_LIBS "opencv_world4100") # Specific OpenCV library names to link, such as opencv_world4100, opencv_java4, etc.
set(ENABLE_NNDEPLOY_OPENCV ON)
set(NNDEPLOY_OPENCV_LIBS) # Link all OpenCV libraries by default

## Tokenizer-cpp
set(ENABLE_NNDEPLOY_PLUGIN_TOKENIZER_CPP OFF) # Whether to enable C++ tokenizer plugin, default is OFF

## Language Model
set(ENABLE_NNDEPLOY_PLUGIN_LLM OFF) # Whether to enable language model plugin, default is OFF

## Stable Diffusion
set(ENABLE_NNDEPLOY_PLUGIN_STABLE_DIFFUSION OFF) # Whether to enable text-to-image plugin, default is OFF
# ---- WSL 本地编译 (x86_64) 特定配置 ----
set(ENABLE_NNDEPLOY_INFERENCE_DEFAULT ON)

# Device 后端（x86_64 本地无需特殊设备）
set(ENABLE_NNDEPLOY_DEVICE_CUDA OFF)
set(ENABLE_NNDEPLOY_DEVICE_OPENCL OFF)
set(ENABLE_NNDEPLOY_DEVICE_VULKAN OFF)

# Operator 后端
set(ENABLE_NNDEPLOY_DEVICE_X86_ONEDNN OFF)

# 推理后端（默认 + 可按需启用）
# set(ENABLE_NNDEPLOY_INFERENCE_ONNXRUNTIME ON)

# ============================================================
# Algorithm plugins — 全部开启（含近 2 个月新增算法插件）
# ============================================================
set(ENABLE_NNDEPLOY_PLUGIN ON)

# 核心基础插件
set(ENABLE_NNDEPLOY_PLUGIN_PREPROCESS ON)
set(ENABLE_NNDEPLOY_PLUGIN_INFER ON)
set(ENABLE_NNDEPLOY_PLUGIN_CODEC ON)

# 轻量级插件（无需额外依赖）
set(ENABLE_NNDEPLOY_PLUGIN_CLASSIFICATION ON)
set(ENABLE_NNDEPLOY_PLUGIN_SUPER_RESOLUTION ON)
set(ENABLE_NNDEPLOY_PLUGIN_OCR ON)
set(ENABLE_NNDEPLOY_PLUGIN_MATTING ON)
set(ENABLE_NNDEPLOY_PLUGIN_MATTING_PPMATTING ON)

# Detect 检测 — YOLO 系列 / RT-DETR / YOLO-NAS / YOLO-OBB / RF-DETR
set(ENABLE_NNDEPLOY_PLUGIN_DETECT ON)
set(ENABLE_NNDEPLOY_PLUGIN_DETECT_DETR ON)
set(ENABLE_NNDEPLOY_PLUGIN_DETECT_YOLO ON)
set(ENABLE_NNDEPLOY_PLUGIN_DETECT_YOLO_OBB ON)
set(ENABLE_NNDEPLOY_PLUGIN_DETECT_RF_DETR ON)

# Segment 分割 — SAM / RMBG / YOLO-Seg / RF-DETR-Seg / SAM2 / SAM3
set(ENABLE_NNDEPLOY_PLUGIN_SEGMENT ON)
set(ENABLE_NNDEPLOY_PLUGIN_SEGMENT_SEGMENT_ANYTHING ON)
set(ENABLE_NNDEPLOY_PLUGIN_SEGMENT_RMBG ON)
set(ENABLE_NNDEPLOY_PLUGIN_SEGMENT_YOLO_SEG ON)
set(ENABLE_NNDEPLOY_PLUGIN_SEGMENT_RF_DETR_SEG ON)
set(ENABLE_NNDEPLOY_PLUGIN_SEGMENT_SAM2 ON)
set(ENABLE_NNDEPLOY_PLUGIN_SEGMENT_SAM3 ON)

# Keypoint 关键点 — YOLO-Pose / RF-DETR-Pose
set(ENABLE_NNDEPLOY_PLUGIN_KEYPOINT ON)
set(ENABLE_NNDEPLOY_PLUGIN_KEYPOINT_YOLO_POSE ON)
set(ENABLE_NNDEPLOY_PLUGIN_KEYPOINT_RF_DETR_POSE ON)

# Track 跟踪 — FairMOT / ByteTrack / BotSORT
set(ENABLE_NNDEPLOY_PLUGIN_TRACK ON)
set(ENABLE_NNDEPLOY_PLUGIN_TRACK_FAIRMOT ON)
set(ENABLE_NNDEPLOY_PLUGIN_TRACK_BYTETRACK ON)
set(ENABLE_NNDEPLOY_PLUGIN_TRACK_BOTSORT ON)
set(ENABLE_NNDEPLOY_PLUGIN_TRACK_BOXMOT ON)


# Depth 深度估计 — Depth Anything 等
set(ENABLE_NNDEPLOY_PLUGIN_DEPTH ON)

# Grounding 开放词汇检测 — GroundingDINO / YOLO-World / YOLOE-Prompt / Florence2 / LocateAnything
set(ENABLE_NNDEPLOY_PLUGIN_GROUNDING ON)
set(ENABLE_NNDEPLOY_PLUGIN_GROUNDING_DINO ON)
set(ENABLE_NNDEPLOY_PLUGIN_GROUNDING_YOLO_WORLD ON)
set(ENABLE_NNDEPLOY_PLUGIN_GROUNDING_YOLOE_PROMPT ON)
set(ENABLE_NNDEPLOY_PLUGIN_GROUNDING_FLORENCE2 OFF)
set(ENABLE_NNDEPLOY_PLUGIN_GROUNDING_FLORENCE2_ONNX OFF)
set(ENABLE_NNDEPLOY_PLUGIN_GROUNDING_LOCATE_ANYTHING OFF)

# Face / GAN / Repair（可选，无额外子插件开关）
set(ENABLE_NNDEPLOY_PLUGIN_FACE ON)
set(ENABLE_NNDEPLOY_PLUGIN_GAN ON)
set(ENABLE_NNDEPLOY_PLUGIN_REPAIR ON)

# 重量级插件（默认关闭，按需启用）
set(ENABLE_NNDEPLOY_PLUGIN_TOKENIZER OFF)
set(ENABLE_NNDEPLOY_PLUGIN_TOKENIZER_CPP OFF)
set(ENABLE_NNDEPLOY_PLUGIN_LLM OFF)
set(ENABLE_NNDEPLOY_PLUGIN_STABLE_DIFFUSION OFF)

# FFmpeg codec backend — disabled for now (build issues with WSL FFmpeg version)
set(ENABLE_NNDEPLOY_FFMPEG OFF)

# Build targets
set(ENABLE_NNDEPLOY_DEMO ON)
set(ENABLE_NNDEPLOY_PYTHON ON)
set(ENABLE_NNDEPLOY_TEST OFF)
set(ENABLE_NNDEPLOY_FFI_JAVA OFF)
