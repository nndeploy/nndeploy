
#ifndef _NNDEPLOY_SOURCE_INFERENCE_PRE_POST_PROCESS_H_
#define _NNDEPLOY_SOURCE_INFERENCE_PRE_POST_PROCESS_H_

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

namespace nndeploy {
namespace inference {

class PrePostProcess {
 public:
  PrePostProcess() {};
  ~PrePostProcess() {};

  base::Status setDevice(device::Device *device);
  device::Device *getDevice();
  device::Device *getDevice(int index);
  device::Device *getDevice(base::DeviceType device_type);

  virtual base::Status setInput(device::Packet &input);
  virtual base::Status setOutput(device::Packet &output);

  virtual base::Status run() = 0;
  virtual base::Status asyncRun() = 0;

protected:
  std::vector<device::Device *> device_;
  device::Packet input_;
  device::Packet output_;
};

}  // namespace inference
}  // namespace nndeploy

#endif