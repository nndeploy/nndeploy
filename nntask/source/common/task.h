
#ifndef _NNTASK_SOURCE_COMMON_TASK_H_
#define _NNTASK_SOURCE_COMMON_TASK_H_

#include "nndeploy/source/base/basic.h"
#include "nndeploy/source/base/glic_stl_include.h"
#include "nndeploy/source/base/log.h"
#include "nndeploy/source/base/macro.h"
#include "nndeploy/source/base/object.h"
#include "nndeploy/source/base/status.h"
#include "nndeploy/source/base/string.h"
#include "nndeploy/source/base/value.h"
#include "nndeploy/source/device/buffer.h"
#include "nndeploy/source/device/buffer_pool.h"
#include "nndeploy/source/device/device.h"
#include "nndeploy/source/device/tensor.h"
#include "nndeploy/source/inference/inference.h"
#include "nndeploy/source/inference/inference_param.h"
#include "nntask/source/common/execution.h"
#include "nntask/source/common/packet.h"

namespace nntask {
namespace common {

class Task : public Execution {
 public:
  Task(nndeploy::base::InferenceType type,
       nndeploy::base::DeviceType device_type, const std::string &name = "");

  virtual ~Task();

  nndeploy::base::Param *getPreProcessParam();
  nndeploy::base::Param *getInferenceParam();
  nndeploy::base::Param *getPostProcessParam();

  virtual nndeploy::base::Status init();
  virtual nndeploy::base::Status deinit();

  virtual nndeploy::base::Status setInput(Packet &input);
  virtual nndeploy::base::Status setOutput(Packet &output);

  virtual nndeploy::base::Status run();

 protected:
  nndeploy::base::InferenceType type_;
  Execution *pre_process_ = nullptr;
  nndeploy::inference::Inference *inference_ = nullptr;
  Execution *post_process_ = nullptr;
};

}  // namespace common
}  // namespace nntask

#endif
