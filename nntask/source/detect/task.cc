#include "nntask/source/detect/task.h"

namespace nntask {
namespace detect {

YoloTask::YoloTask(bool allcoate_tensor_flag,
                   nndeploy::base::InferenceType type,
                   nndeploy::base::DeviceType device_type,
                   const std::string& name)
    : common::StaticShapeTask(allcoate_tensor_flag, type, device_type, name) {
  pre_process_ = new common::opencv::CvtcolorResize(device_type);
  CvtclorResizeParam* pre_param =
      dynamic_cast<CvtclorResizeParam*>(pre_process_->getParam());
  pre_param->src_pix_type_ = nndeploy::base::kPixelTypeBGR;
  pre_param->dst_pix_type_ = nndeploy::base::kPixelTypeBGR;
  pre_param->interp_type_ = nndeploy::base::kInterpTypeLinear;
  pre_param->scale_.push_back(1 / 255.0f);
  pre_param->scale_.push_back(1 / 255.0f);
  pre_param->scale_.push_back(1 / 255.0f);
  pre_param->scale_.push_back(1 / 255.0f);
  pre_param->bias_.push_back(0.0f);
  pre_param->bias_.push_back(0.0f);
  pre_param->bias_.push_back(0.0f);
  pre_param->bias_.push_back(0.0f);

  post_process_ = new common::opencv::DetectPostProcess(device_type);
}

}  // namespace detect
}  // namespace nntask