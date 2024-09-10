-- framework
option("ENABLE_NNDEPLOY_BUILD_SHARED")
    set_default(true)

option("ENABLE_NNDEPLOY_SYMBOL_HIDE")
    set_default(false)
    
option("ENABLE_NNDEPLOY_COVERAGE")
    set_default(false)

option("ENABLE_NNDEPLOY_CXX11_ABI")
    set_default(true)

option("ENABLE_NNDEPLOY_CXX14_ABI")
    set_default(false)

option("ENABLE_NNDEPLOY_CXX17_ABI")
    set_default(false)

option("ENABLE_NNDEPLOY_CXX20_ABI")
    set_default(false)

option("ENABLE_NNDEPLOY_OPENMP")
    set_default(false)

option("ENABLE_NNDEPLOY_ADDRESS_SANTIZER")
    set_default(false)

option("ENABLE_NNDEPLOY_DOCS")
    set_default(false)

option("ENABLE_NNDEPLOY_TIME_PROFILER")
    set_default(true)

option("ENABLE_NNDEPLOY_OPENCV")
    set_default(true)

-- base
option("ENABLE_NNDEPLOY_BASE")
    set_default(true)

option("ENABLE_NNDEPLOY_THREAD_POOL")
    set_default(true)

-- cryption
option("ENABLE_NNDEPLOY_CRYPTION")
    set_default(false)

-- device
option("ENABLE_NNDEPLOY_DEVICE")
    set_default(true)

option("ENABLE_NNDEPLOY_DEVICE_CPU")
    set_default(true)

option("ENABLE_NNDEPLOY_DEVICE_ARM")    
    set_default(false)

option("ENABLE_NNDEPLOY_DEVICE_X86")
    set_default(false)

option("ENABLE_NNDEPLOY_DEVICE_CUDA")
    set_default(false)

option("ENABLE_NNDEPLOY_DEVICE_CUDNN")
    set_default(false)

option("ENABLE_NNDEPLOY_DEVICE_OPENCL")
    set_default(false)

option("ENABLE_NNDEPLOY_DEVICE_OPENGL")
    set_default(false)

option("ENABLE_NNDEPLOY_DEVICE_METAL")
    set_default(false)

option("ENABLE_NNDEPLOY_DEVICE_APPLE_NPU")
    set_default(false)

option("ENABLE_NNDEPLOY_DEVICE_HVX")
    set_default(false)

option("ENABLE_NNDEPLOY_DEVICE_MTK_VPU")
    set_default(false)

option("ENABLE_NNDEPLOY_DEVICE_ASCEND_CL")
    set_default(false)

-- op
option("ENABLE_NNDEPLOY_OP")
    set_default(false)

-- interpret
option("ENABLE_NNDEPLOY_INTERPRET")
    set_default(false)

-- net
option("ENABLE_NNDEPLOY_NET")
    set_default(false)

-- inference
option("ENABLE_NNDEPLOY_INFERENCE")
    set_default(true)

option("ENABLE_NNDEPLOY_INFERENCE_TENSORRT")
    set_default(false)

option("ENABLE_NNDEPLOY_INFERENCE_OPENVINO")
    set_default(false)

option("ENABLE_NNDEPLOY_INFERENCE_COREML")
    set_default(false)

option("ENABLE_NNDEPLOY_INFERENCE_TFLITE")
    set_default(false)

option("ENABLE_NNDEPLOY_INFERENCE_ONNXRUNTIME")
    set_default(true)

option("ENABLE_NNDEPLOY_INFERENCE_NCNN")
    set_default(false)

option("ENABLE_NNDEPLOY_INFERENCE_TNN")
    set_default(false)

option("ENABLE_NNDEPLOY_INFERENCE_MNN")
    set_default(false)

option("ENABLE_NNDEPLOY_INFERENCE_TVM")
    set_default(false)

option("ENABLE_NNDEPLOY_INFERENCE_PADDLELITE")
    set_default(false)

option("ENABLE_NNDEPLOY_INFERENCE_RKNN_TOOLKIT_1")
    set_default(false)

option("ENABLE_NNDEPLOY_INFERENCE_RKNN_TOOLKIT_2")
    set_default(false)

option("ENABLE_NNDEPLOY_INFERENCE_ASCEND_CL")
    set_default(false)

option("ENABLE_NNDEPLOY_INFERENCE_SNPE")
    set_default(false)

-- dag
option("ENABLE_NNDEPLOY_DAG")
    set_default(true)

-- plugin
option("ENABLE_NNDEPLOY_PLUGIN")
    set_default(false)

-- test
option("ENABLE_NNDEPLOY_TEST")
    set_default(false)

-- demo
option("ENABLE_NNDEPLOY_DEMO")
    set_default(false)

 