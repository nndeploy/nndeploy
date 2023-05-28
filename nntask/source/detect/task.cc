#include "nntask/source/detect/task.h"

namespace nntask {
namespace detect {

YoloTask::YoloTask(bool allcoate_tensor_flag,
                   nndeploy::base::InferenceType type,
                   nndeploy::base::DeviceType device_type,
                   const std::string& name)
    : common::StaticShapeTask(allcoate_tensor_flag, type, device_type, name) {
  pre_process_ = new common::OpencvCvtColrResize(device_type);
  common::CvtclorResizeParam* pre_param =
      dynamic_cast<common::CvtclorResizeParam*>(pre_process_->getParam());
  pre_param->src_pixel_type_ = nndeploy::base::kPixelTypeBGR;
  pre_param->dst_pixel_type_ = nndeploy::base::kPixelTypeBGR;
  pre_param->interp_type_ = nndeploy::base::kInterpTypeLinear;
  pre_param->mean_[0] = 0.0f;
  pre_param->mean_[1] = 0.0f;
  pre_param->mean_[2] = 0.0f;
  pre_param->mean_[3] = 0.0f;
  pre_param->std_[0] = 255.0f;
  pre_param->std_[1] = 255.0f;
  pre_param->std_[2] = 255.0f;
  pre_param->std_[3] = 255.0f;

  post_process_ = new common::DetectPostProcess(device_type);
}

// DetrTask::DetrTask(bool allcoate_tensor_flag,
//                    nndeploy::base::InferenceType type,
//                    nndeploy::base::DeviceType device_type,
//                    const std::string& name)
//     : common::StaticShapeTask(allcoate_tensor_flag, type, device_type, name)
//     {
//   pre_process_ = new DetrPreProcess(device_type);

//   post_process_ = new DetrPostProcess(device_type);
// }

}  // namespace detect
}  // namespace nntask