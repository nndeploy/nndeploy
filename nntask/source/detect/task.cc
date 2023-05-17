#include "nntask/source/detect/task.h"

#include "nndeploy/source/base/glic_stl_include.h"

namespace nntask {
namespace detect {

Task::Task(bool allcoate_tensor_flag, nndeploy::base::InferenceType type,
           nndeploy::base::DeviceType device_type, const std::string &name)
    : StaticShape(allcoate_tensor_flag, type, device_type, name) {
  process = new common::ResizeBn();
  post_process = new common::detect();
}

}  // namespace detect
}  // namespace nntask