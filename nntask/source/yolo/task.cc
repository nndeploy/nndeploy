#include "nntask/source/yolo/task.h"

#include "nndeploy/source/base/glic_stl_include.h"

namespace nntask {
namespace yolo {

Task::Task(bool allcoate_tensor_flag, nndeploy::base::InferenceType type,
           nndeploy::base::DeviceType device_type, const std::string &name)
    : StaticShape(allcoate_tensor_flag, type, device_type, name) {}

}  // namespace yolo
}  // namespace nntask