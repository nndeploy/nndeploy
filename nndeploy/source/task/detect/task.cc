#include "nndeploy/source/task/detect/task.h"

namespace nndeploy {
namespace task {

// YoloTask::YoloTask(bool allcoate_tensor_flag, base::InferenceType type,
//                    base::DeviceType device_type, const std::string& name)
//     : StaticShapeTask(allcoate_tensor_flag, type, device_type, name)
//     {
//   pre_process_ = new OpencvCvtColrResize(device_type);
//   CvtclorResizeParam* pre_param =
//       dynamic_cast<CvtclorResizeParam*>(pre_process_->getParam());
//   pre_param->src_pixel_type_ = base::kPixelTypeBGR;
//   pre_param->dst_pixel_type_ = base::kPixelTypeBGR;
//   pre_param->interp_type_ = base::kInterpTypeLinear;
//   pre_param->mean_[0] = 0.0f;
//   pre_param->mean_[1] = 0.0f;
//   pre_param->mean_[2] = 0.0f;
//   pre_param->mean_[3] = 0.0f;
//   pre_param->std_[0] = 255.0f;
//   pre_param->std_[1] = 255.0f;
//   pre_param->std_[2] = 255.0f;
//   pre_param->std_[3] = 255.0f;

//   post_process_ = new DetectPostProcess(device_type);
// }

// DetrTask::DetrTask(bool allcoate_tensor_flag,
//                    base::InferenceType type,
//                    base::DeviceType device_type,
//                    const std::string& name)
//     : StaticShapeTask(allcoate_tensor_flag, type, device_type, name)
//     {
//   pre_process_ = new DetrPreProcess(device_type);

//   post_process_ = new DetrPostProcess(device_type);
// }

}  // namespace task
}  // namespace nndeploy