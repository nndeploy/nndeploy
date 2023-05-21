#include "nntask/source/detect/task.h"

#include "nndeploy/source/base/glic_stl_include.h"
#include "nntask/source/common/process/opencv/pre_process.h"

namespace nntask {
namespace detect {

Task::Task(bool allcoate_tensor_flag, nndeploy::base::InferenceType type,
           nndeploy::base::DeviceType device_type, const std::string &name)
    : StaticShape(allcoate_tensor_flag, type, device_type, name) {
  pre_process_ = new common::OpenCVResizeNorm(device_type);
  post_process_ = new common::DetectPostProcess(device_type);
}

}  // namespace detect
}  // namespace nntask