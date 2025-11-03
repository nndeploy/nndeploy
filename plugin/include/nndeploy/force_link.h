
#ifndef _NNDEPLOY_PLUGIN_FORCE_LINK_H_
#define _NNDEPLOY_PLUGIN_FORCE_LINK_H_

#include "nndeploy/base/macro.h"

#ifdef _MSC_VER

#pragma comment(lib, "nndeploy_plugin_basic.lib")
#include "nndeploy/basic/end.h"
NNDEPLOY_FORCE_LOAD_LIB_SYMBOL(nndeploy::basic::End);

#ifdef ENABLE_NNDEPLOY_PLUGIN_PREPROCESS
#pragma comment(lib, "nndeploy_plugin_preprocess.lib")
#include "nndeploy/preprocess/convert_to.h"
NNDEPLOY_FORCE_LOAD_LIB_SYMBOL(nndeploy::preprocess::ConvertTo);
#endif

#ifdef ENABLE_NNDEPLOY_PLUGIN_INFER
#pragma comment(lib, "nndeploy_plugin_infer.lib")
#include "nndeploy/infer/infer.h"
NNDEPLOY_FORCE_LOAD_LIB_SYMBOL(nndeploy::infer::Infer);
#endif

#ifdef ENABLE_NNDEPLOY_PLUGIN_CODEC
#pragma comment(lib, "nndeploy_plugin_codec.lib")
#include "nndeploy/codec/opencv/opencv_codec.h"
NNDEPLOY_FORCE_LOAD_LIB_SYMBOL(nndeploy::codec::OpenCvImageDecode);
#endif

#ifdef ENABLE_NNDEPLOY_PLUGIN_TOKENIZER
#pragma comment(lib, "nndeploy_plugin_tokenizer.lib")
#include "nndeploy/tokenizer/tokenizer.h"
NNDEPLOY_FORCE_LOAD_LIB_SYMBOL(nndeploy::tokenizer::TokenizerEncode);
#endif

#ifdef ENABLE_NNDEPLOY_PLUGIN_CLASSIFICATION
#pragma comment(lib, "nndeploy_plugin_classification.lib")
#include "nndeploy/classification/classification.h"
NNDEPLOY_FORCE_LOAD_LIB_SYMBOL(
    nndeploy::classification::ClassificationPostProcess);
#endif

#ifdef ENABLE_NNDEPLOY_PLUGIN_LLM
#pragma comment(lib, "nndeploy_plugin_llm.lib")
#pragma comment(lib, "nndeploy_plugin_qwen.lib")
#include "nndeploy/llm/llm_infer.h"
NNDEPLOY_FORCE_LOAD_LIB_SYMBOL(nndeploy::llm::LlmInfer);
#include "nndeploy/llm/decode.h"
NNDEPLOY_FORCE_LOAD_LIB_SYMBOL(nndeploy::llm::Decode);
#include "nndeploy/qwen/qwen.h"
NNDEPLOY_FORCE_LOAD_LIB_SYMBOL(nndeploy::qwen::PromptParam);
#endif

#ifdef ENABLE_NNDEPLOY_PLUGIN_DETECT
#pragma comment(lib, "nndeploy_plugin_detect.lib")
#include "nndeploy/detect/drawbox.h"
NNDEPLOY_FORCE_LOAD_LIB_SYMBOL(nndeploy::detect::DrawBox);
#include "nndeploy/detect/yolo/yolo.h"
NNDEPLOY_FORCE_LOAD_LIB_SYMBOL(nndeploy::detect::YoloPostProcess);
#endif

#ifdef ENABLE_NNDEPLOY_PLUGIN_SEGMENT
#pragma comment(lib, "nndeploy_plugin_segment.lib")
#include "nndeploy/segment/drawmask.h"
NNDEPLOY_FORCE_LOAD_LIB_SYMBOL(nndeploy::segment::DrawMask);
#endif

#ifdef ENABLE_NNDEPLOY_PLUGIN_STABLE_DIFFUSION
#pragma comment(lib, "nndeploy_plugin_stable_diffusion.lib")
#include "nndeploy/stable_diffusion/scheduler.h"
NNDEPLOY_FORCE_LOAD_LIB_SYMBOL(nndeploy::stable_diffusion::SchedulerParam);
#endif

#ifdef ENABLE_NNDEPLOY_PLUGIN_SUPER_RESOLUTION
#pragma comment(lib, "nndeploy_plugin_super_resolution.lib")
#include "nndeploy/super_resolution/super_resolution.h"
NNDEPLOY_FORCE_LOAD_LIB_SYMBOL(
    nndeploy::super_resolution::SuperResolutionPostProcess);
#endif

#ifdef ENABLE_NNDEPLOY_PLUGIN_TRACK
#pragma comment(lib, "nndeploy_plugin_track.lib")
#include "nndeploy/track/fairmot/fairmot.h"
NNDEPLOY_FORCE_LOAD_LIB_SYMBOL(nndeploy::track::FairMotPreParam);
#endif

#ifdef ENABLE_NNDEPLOY_PLUGIN_MATTING
#pragma comment(lib, "nndeploy_plugin_matting.lib")
#include "nndeploy/matting/pp_matting/pp_matting.h"
NNDEPLOY_FORCE_LOAD_LIB_SYMBOL(nndeploy::matting::PPMattingPostParam);
#endif

#ifdef ENABLE_NNDEPLOY_PLUGIN_OCR
#pragma comment(lib, "nndeploy_plugin_ocr.lib")
#include "nndeploy/ocr/result.h"
NNDEPLOY_FORCE_LOAD_LIB_SYMBOL(nndeploy::ocr::OCRResult);
#endif

// ... other plugins
#endif  // _MSC_VER

#endif // _NNDEPLOY_PLUGIN_FORCE_LINK_H_