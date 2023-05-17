#ifndef _NNTASK_SOURCE_DETECT_TASK_H_
#define _NNTASK_SOURCE_DETECT_TASK_H_

#include "nntask/source/common/process//opencv/pre_process.h"
#include "nntask/source/common/process/opencv/detect.h"
#include "nntask/source/common/template_inference_task/static_shape.h"

namespace nntask {
namespace detect {

class Task : public common::StaticShape {
 public:
  Task(bool allcoate_tensor_flag, nndeploy::base::InferenceType type,
       nndeploy::base::DeviceType device_type, const std::string &name);
  virtual ~Task(){};
};

}  // namespace detect
}  // namespace nntask

#endif /* _NNTASK_SOURCE_DETECT_TASK_H_ */
