#ifndef _NNTASK_SOURCE_DETECT_YOLO_TASK_H_
#define _NNTASK_SOURCE_DETECT_YOLO_TASK_H_

#include "nntask/source/common/pre_process/opencv/cvtcolor_resize.h"
#include "nntask/source/common/template_inference_task/static_shape_task.h"
#include "nntask/source/detect/post_process/opencv/post_process.h"

namespace nntask {
namespace detect {

class YoloTask : public common::StaticShapeTask {
 public:
  YoloTask(bool allcoate_tensor_flag, nndeploy::base::InferenceType type,
           nndeploy::base::DeviceType device_type, const std::string &name);
  virtual ~YoloTask(){};
};

class DETRTask : public common::StaticShapeTask {
 public:
  DETRTask(bool allcoate_tensor_flag, nndeploy::base::InferenceType type,
           nndeploy::base::DeviceType device_type, const std::string &name);
  virtual ~DETRTask(){};
};


}  // namespace detect
}  // namespace nntask

#endif /* _NNTASK_SOURCE_DETECT_YOLO_H_ */
