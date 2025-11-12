

#include "nndeploy/base/macro.h"

#include "nndeploy/basic/end.h"
NNDEPLOY_FORCE_LOAD_LIB_SYMBOL(nndeploy::basic::End);

#ifdef ENABLE_NNDEPLOY_PLUGIN_PREPROCESS
#include "nndeploy/preprocess/convert_to.h"
NNDEPLOY_FORCE_LOAD_LIB_SYMBOL(nndeploy::preprocess::ConvertTo);
#endif

#ifdef ENABLE_NNDEPLOY_PLUGIN_INFER
#include "nndeploy/infer/infer.h"
NNDEPLOY_FORCE_LOAD_LIB_SYMBOL(nndeploy::infer::Infer);
#endif

#ifdef ENABLE_NNDEPLOY_PLUGIN_CODEC
#include "nndeploy/codec/opencv/opencv_codec.h"
NNDEPLOY_FORCE_LOAD_LIB_SYMBOL(nndeploy::codec::OpenCvImageDecode);
#endif

#ifdef ENABLE_NNDEPLOY_PLUGIN_TOKENIZER
#include "nndeploy/tokenizer/tokenizer.h"
NNDEPLOY_FORCE_LOAD_LIB_SYMBOL(nndeploy::tokenizer::TokenizerEncode);
#endif

#ifdef ENABLE_NNDEPLOY_PLUGIN_CLASSIFICATION
#include "nndeploy/classification/classification.h"
NNDEPLOY_FORCE_LOAD_LIB_SYMBOL(
    nndeploy::classification::ClassificationPostProcess);
#endif

#ifdef ENABLE_NNDEPLOY_PLUGIN_LLM
#include "nndeploy/llm/llm_infer.h"
NNDEPLOY_FORCE_LOAD_LIB_SYMBOL(nndeploy::llm::LlmInfer);
#include "nndeploy/llm/decode.h"
NNDEPLOY_FORCE_LOAD_LIB_SYMBOL(nndeploy::llm::Decode);
#include "nndeploy/qwen/qwen.h"
NNDEPLOY_FORCE_LOAD_LIB_SYMBOL(nndeploy::qwen::PromptParam);
#endif

#ifdef ENABLE_NNDEPLOY_PLUGIN_DETECT
#include "nndeploy/detect/drawbox.h"
NNDEPLOY_FORCE_LOAD_LIB_SYMBOL(nndeploy::detect::DrawBox);
#include "nndeploy/detect/yolo/yolo.h"
NNDEPLOY_FORCE_LOAD_LIB_SYMBOL(nndeploy::detect::YoloPostProcess);
#endif

#ifdef ENABLE_NNDEPLOY_PLUGIN_SEGMENT
#include "nndeploy/segment/drawmask.h"
NNDEPLOY_FORCE_LOAD_LIB_SYMBOL(nndeploy::segment::DrawMask);
#endif

#ifdef ENABLE_NNDEPLOY_PLUGIN_STABLE_DIFFUSION
#include "nndeploy/stable_diffusion/scheduler.h"
NNDEPLOY_FORCE_LOAD_LIB_SYMBOL(nndeploy::stable_diffusion::SchedulerParam);
#endif

#ifdef ENABLE_NNDEPLOY_PLUGIN_SUPER_RESOLUTION
#include "nndeploy/super_resolution/super_resolution.h"
NNDEPLOY_FORCE_LOAD_LIB_SYMBOL(
    nndeploy::super_resolution::SuperResolutionPostProcess);
#endif

#ifdef ENABLE_NNDEPLOY_PLUGIN_TRACK
#include "nndeploy/track/fairmot/fairmot.h"
NNDEPLOY_FORCE_LOAD_LIB_SYMBOL(nndeploy::track::FairMotPreParam);
#endif

#ifdef ENABLE_NNDEPLOY_PLUGIN_MATTING
#include "nndeploy/matting/pp_matting/pp_matting.h"
NNDEPLOY_FORCE_LOAD_LIB_SYMBOL(nndeploy::matting::PPMattingPostParam);
#endif

#ifdef ENABLE_NNDEPLOY_PLUGIN_OCR
#include "nndeploy/ocr/result.h"
NNDEPLOY_FORCE_LOAD_LIB_SYMBOL(nndeploy::ocr::OCRResult);
#endif

// ... other plugins