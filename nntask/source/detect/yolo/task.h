#ifndef DBC94EFC_B1B0_42DF_88BE_9CC021FE3A5E
#define DBC94EFC_B1B0_42DF_88BE_9CC021FE3A5E

#include "nntask/source/common/post_process/yolo.h"
#include "nntask/source/common/pre_process/resize_bn.h"
#include "nntask/source/common/template_inference_task/static_shape.h"

namespace nntask {
namespace yolo {

class Task : public common::StaticShape {
 public:
  Task(bool allcoate_tensor_flag, nndeploy::base::InferenceType type,
       nndeploy::base::DeviceType device_type, const std::string &name);
  virtual ~Task(){};
};

}  // namespace yolo
}  // namespace nntask

#endif /* DBC94EFC_B1B0_42DF_88BE_9CC021FE3A5E */
