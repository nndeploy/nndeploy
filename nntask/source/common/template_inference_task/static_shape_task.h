
#ifndef _NNTASK_SOURCE_COMMON_TEMPLATE_INFERENCE_TASK_STATIC_SHAPE_TASK_H_
#define _NNTASK_SOURCE_COMMON_TEMPLATE_INFERENCE_TASK_STATIC_SHAPE_TASK_H_

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
#include "nntask/source/common/task.h"

namespace nntask {
namespace common {

class StaticShapeTask : public Task {
 public:
  StaticShapeTask(bool allcoate_tensor_flag, nndeploy::base::InferenceType type,
                  nndeploy::base::DeviceType device_type,
                  const std::string &name = "");

  virtual ~StaticShapeTask();

  virtual nndeploy::base::Status init();
  virtual nndeploy::base::Status deinit();

  virtual nndeploy::base::Status setInput(Packet &input);
  virtual nndeploy::base::Status setOutput(Packet &output);

  virtual nndeploy::base::Status run();

 private:
  nndeploy::base::Status allocateInputOutputTensor();
  nndeploy::base::Status deallocateInputOutputTensor();

 protected:
  bool allcoate_tensor_flag_ = false;
  std::vector<nndeploy::device::Tensor *> input_tensors_;
  Packet *inference_input_packet_;
  std::vector<nndeploy::device::Tensor *> output_tensors_;
  Packet *inference_output_packet_;
};

}  // namespace common
}  // namespace nntask

#endif
