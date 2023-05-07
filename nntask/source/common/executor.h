
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
#include "nndeploy/source/inference/inference.h"
#include "nndeploy/source/inference/inference_param.h"
#include "nntask/source/common/packet.h"

namespace nntask {
namespace common {

class Executor {
 public:
  Executor(std::string name = "");
  ~Executor();

  virtual nndeploy::base::Param* getParam();

  virtual nndeploy::base::Status init();
  virtual nndeploy::base::Status deinit();

  virtual nndeploy::base::Status setInput(Packet& input);
  virtual nndeploy::base::Status setOutput(Packet& output);

  virtual nndeploy::base::Status run() = 0;

 protected:
  std::string name_ = "";
  nndeploy::base::Param* param_ = nullptr;
  Packet* input_ = nullptr;
  Packet* output_ = nullptr;
};

}  // namespace common
}  // namespace nntask

#endif