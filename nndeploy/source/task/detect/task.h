#ifndef _NNTASK_SOURCE_DETECT_YOLO_TASK_H_
#define _NNTASK_SOURCE_DETECT_YOLO_TASK_H_

#include "nndeploy/source/task/detect/opencv/post_process.h"
#include "nndeploy/source/task/pre_process/opencv/cvtcolor_resize.h"
#include "nndeploy/source/task/task.h"

namespace nndeploy {
namespace task {

// class YoloTask : public common::StaticShapeTask {
//  public:
//   YoloTask(bool allcoate_tensor_flag, base::InferenceType type,
//            base::DeviceType device_type, const std::string &name);
//   virtual ~YoloTask(){};
// };

// class DetrTask : public common::StaticShapeTask {
//  public:
//   DetrTask(bool allcoate_tensor_flag, base::InferenceType type,
//            base::DeviceType device_type, const std::string &name);
//   virtual ~DetrTask(){};
// };

}  // namespace task
}  // namespace nndeploy

#endif /* _NNTASK_SOURCE_DETECT_YOLO_H_ */
