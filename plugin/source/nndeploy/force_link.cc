

#include "nndeploy/base/macro.h"
#include "nndeploy/basic/end.h"
NNDEPLOY_FORCE_LOAD_LIB_SYMBOL(nndeploy::basic::End);

#ifdef ENABLE_NNDEPLOY_PLUGIN_PREPROCESS
#include "nndeploy/preprocess/convert_to.h"
NNDEPLOY_FORCE_LOAD_LIB_SYMBOL(nndeploy::preprocess::ConvertTo);
#include "nndeploy/preprocess/cvt_norm_trans.h"
NNDEPLOY_FORCE_LOAD_LIB_SYMBOL(nndeploy::preprocess::CvtNormTrans);
#include "nndeploy/preprocess/cvt_resize_norm_trans.h"
NNDEPLOY_FORCE_LOAD_LIB_SYMBOL(nndeploy::preprocess::CvtResizeNormTrans);
#include "nndeploy/preprocess/cvt_resize_crop_norm_trans.h"
NNDEPLOY_FORCE_LOAD_LIB_SYMBOL(nndeploy::preprocess::CvtResizeCropNormTrans);
#include "nndeploy/preprocess/cvt_resize_pad_norm_trans.h"
NNDEPLOY_FORCE_LOAD_LIB_SYMBOL(nndeploy::preprocess::CvtResizePadNormTrans);
#include "nndeploy/preprocess/warp_affine_cvt_norm_trans.h"
NNDEPLOY_FORCE_LOAD_LIB_SYMBOL(nndeploy::preprocess::WarpAffineCvtNormTrans);
#endif

#ifdef ENABLE_NNDEPLOY_RKRGA
#include "nndeploy/preprocess/rga/rga_cvt_resize_norm_trans.h"
NNDEPLOY_FORCE_LOAD_LIB_SYMBOL(nndeploy::preprocess::RgaCvtResizeNormTrans);
#include "nndeploy/preprocess/rga/rga_cvt_color.h"
NNDEPLOY_FORCE_LOAD_LIB_SYMBOL(nndeploy::preprocess::RgaCvtColor);
#include "nndeploy/preprocess/rga/rga_resize.h"
NNDEPLOY_FORCE_LOAD_LIB_SYMBOL(nndeploy::preprocess::RgaResize);
#include "nndeploy/preprocess/rga/rga_dma_buf_to_mat.h"
NNDEPLOY_FORCE_LOAD_LIB_SYMBOL(nndeploy::preprocess::DmaBufToMat);
#include "nndeploy/preprocess/rga/rga_dma_buf_to_tensor.h"
NNDEPLOY_FORCE_LOAD_LIB_SYMBOL(nndeploy::preprocess::RgaDmaBufToTensor);
#include "nndeploy/preprocess/rga/rga_crop.h"
NNDEPLOY_FORCE_LOAD_LIB_SYMBOL(nndeploy::preprocess::RgaCrop);
#include "nndeploy/preprocess/rga/rga_rotate.h"
NNDEPLOY_FORCE_LOAD_LIB_SYMBOL(nndeploy::preprocess::RgaRotate);
#include "nndeploy/preprocess/rga/rga_flip.h"
NNDEPLOY_FORCE_LOAD_LIB_SYMBOL(nndeploy::preprocess::RgaFlip);
#include "nndeploy/preprocess/rga/rga_letterbox.h"
NNDEPLOY_FORCE_LOAD_LIB_SYMBOL(nndeploy::preprocess::RgaLetterbox);
#include "nndeploy/preprocess/rga/rga_cvt_resize_pad_norm_trans.h"
NNDEPLOY_FORCE_LOAD_LIB_SYMBOL(nndeploy::preprocess::RgaCvtResizePadNormTrans);
#include "nndeploy/preprocess/rga/rga_cvt_resize_crop_norm_trans.h"
NNDEPLOY_FORCE_LOAD_LIB_SYMBOL(nndeploy::preprocess::RgaCvtResizeCropNormTrans);
#include "nndeploy/preprocess/rga/rga_fuse_pass.h"
NNDEPLOY_FORCE_LOAD_LIB_SYMBOL(nndeploy::dag::RgaFusePass);
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
NNDEPLOY_FORCE_LOAD_LIB_SYMBOL(nndeploy::detect::DrawBBox);
#include "nndeploy/detect/yolo/yolo.h"
NNDEPLOY_FORCE_LOAD_LIB_SYMBOL(nndeploy::detect::YoloPostProcess);
#endif

#ifdef ENABLE_NNDEPLOY_PLUGIN_SEGMENT
#include "nndeploy/segment/drawmask.h"
NNDEPLOY_FORCE_LOAD_LIB_SYMBOL(nndeploy::segment::DrawMask);
NNDEPLOY_FORCE_LOAD_LIB_SYMBOL(nndeploy::segment::DrawSegMask);
#ifdef ENABLE_NNDEPLOY_PLUGIN_SEGMENT_SEGMENT_ANYTHING
#include "nndeploy/segment/segment_anything/sam.h"
NNDEPLOY_FORCE_LOAD_LIB_SYMBOL(nndeploy::segment::SAMGraph);
#endif
#ifdef ENABLE_NNDEPLOY_PLUGIN_SEGMENT_SAM2
#include "nndeploy/segment/segment_anything/sam2.h"
NNDEPLOY_FORCE_LOAD_LIB_SYMBOL(nndeploy::segment::SAM2Graph);
#endif
#ifdef ENABLE_NNDEPLOY_PLUGIN_SEGMENT_SAM3
#include "nndeploy/segment/segment_anything/sam3.h"
NNDEPLOY_FORCE_LOAD_LIB_SYMBOL(nndeploy::segment::SAM3Graph);
#endif
#ifdef ENABLE_NNDEPLOY_PLUGIN_SEGMENT_RF_DETR_SEG
#include "nndeploy/segment/rf_detr_seg/rf_detr_seg.h"
NNDEPLOY_FORCE_LOAD_LIB_SYMBOL(nndeploy::segment::RfDetrSegPostProcess);
NNDEPLOY_FORCE_LOAD_LIB_SYMBOL(nndeploy::segment::RfDetrSegGraph);
#endif
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
#ifdef ENABLE_NNDEPLOY_PLUGIN_TRACK_BOXMOT
#include "nndeploy/track/boxmot/result.h"
NNDEPLOY_FORCE_LOAD_LIB_SYMBOL(nndeploy::track::BoxMotParam);
#include "nndeploy/track/boxmot/boxmot_node.h"
NNDEPLOY_FORCE_LOAD_LIB_SYMBOL(nndeploy::track::BoxMotNode);
#endif
#endif

#ifdef ENABLE_NNDEPLOY_PLUGIN_MATTING
#include "nndeploy/matting/pp_matting/pp_matting.h"
NNDEPLOY_FORCE_LOAD_LIB_SYMBOL(nndeploy::matting::PPMattingPostParam);
#endif

#ifdef ENABLE_NNDEPLOY_PLUGIN_OCR
#include "nndeploy/ocr/result.h"
NNDEPLOY_FORCE_LOAD_LIB_SYMBOL(nndeploy::ocr::OCRResult);
#endif

#ifdef ENABLE_NNDEPLOY_PLUGIN_KEYPOINT
#include "nndeploy/keypoint/result.h"
NNDEPLOY_FORCE_LOAD_LIB_SYMBOL(nndeploy::keypoint::KeypointResult);
#ifdef ENABLE_NNDEPLOY_PLUGIN_KEYPOINT_YOLO_POSE
#include "nndeploy/keypoint/yolo_pose/yolo_pose.h"
NNDEPLOY_FORCE_LOAD_LIB_SYMBOL(nndeploy::keypoint::KeypointPostProcess);
#endif
#endif

#ifdef ENABLE_NNDEPLOY_PLUGIN_GROUNDING
#include "nndeploy/grounding/result.h"
NNDEPLOY_FORCE_LOAD_LIB_SYMBOL(nndeploy::grounding::DetectResult);
#ifdef ENABLE_NNDEPLOY_PLUGIN_GROUNDING_DINO
#include "nndeploy/grounding/grounding_dino.h"
NNDEPLOY_FORCE_LOAD_LIB_SYMBOL(nndeploy::grounding::GroundingDINOGraph);
#endif
#ifdef ENABLE_NNDEPLOY_PLUGIN_GROUNDING_YOLO_WORLD
#include "nndeploy/grounding/yolo_world/yolo_world.h"
NNDEPLOY_FORCE_LOAD_LIB_SYMBOL(nndeploy::grounding::YoloWorldGraph);
#include "nndeploy/grounding/yolo_world/clip_text_encode.h"
NNDEPLOY_FORCE_LOAD_LIB_SYMBOL(nndeploy::grounding::ClipTextEncode);
NNDEPLOY_FORCE_LOAD_LIB_SYMBOL(nndeploy::grounding::TokenIdsToTensor);
NNDEPLOY_FORCE_LOAD_LIB_SYMBOL(nndeploy::grounding::L2Normalize);
#endif
#endif
