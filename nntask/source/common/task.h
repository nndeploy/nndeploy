
#ifndef _NNDEPLOY_SOURCE_TASKFLOW_TASK_H_
#define _NNDEPLOY_SOURCE_TASKFLOW_TASK_H_

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
#include "nndeploy/source/inference/abstract_inference_impl.h"
#include "nndeploy/source/inference/config.h"
#include "nndeploy/source/device/packet.h"
#include "nndeploy/source/inference/inference.h"
#include "nndeploy/source/inference/pre_post_process.h"

namespace nndeploy {
namespace taskflow {

class Task {
 public:
  Task() {};
  Task(base::InferenceType type) {};
  ~Task() {};

  virtual base::Status setDevice(device::Device *device);
  virtual device::Device *getDevice();
  virtual device::Device *getDevice(int index);
  virtual device::Device *getDevice(base::DeviceType device_type);

  virtual base::Status init(std::shared_ptr<inference::Config> config);
  virtual base::Status deinit();

  virtual base::Status preRun(base::ShapeMap min_shape = base::ShapeMap(),
                      base::ShapeMap opt_shape = base::ShapeMap(),
                      base::ShapeMap max_shape = base::ShapeMap());
  virtual base::Status postRun();

  virtual base::Status setInput(device::Packet &input);
  virtual base::Status setOutput(device::Packet &output);

  virtual base::Status run();
  virtual base::Status asyncRun();

protected:
  std::vector<device::Device *> device_;
  device::Packet input_;
  device::Packet output_;

  bool is_constract_ = true;
  inference::PrePostProcess *pre_processs_ = nullptr;
  inference::Inference *inference_ = nullptr;
  inference::PrePostProcess *post_processs_= nullptr;
};

}  // namespace inference
}  // namespace nndeploy

#endif
