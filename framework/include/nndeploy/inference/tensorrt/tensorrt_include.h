
#ifndef _NNDEPLOY_INFERENCE_TENSORRT_TENSORRT_INCLUDE_H_
#define _NNDEPLOY_INFERENCE_TENSORRT_TENSORRT_INCLUDE_H_

#include <NvInfer.h>
#include <NvInferPlugin.h>
#include <NvInferRuntime.h>
#include <NvInferVersion.h>
#include <NvOnnxParser.h>

#include "nndeploy/device/cuda/cuda_include.h"

#if NV_TENSORRT_MAJOR > 7 && NV_TENSORRT_MINOR > 4
#define TENSORRT_MAJOR_8_MINOR_5
#endif

#endif