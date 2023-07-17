
#ifndef _NNDEPLOY_PIPELINE_TASK_H_
#define _NNDEPLOY_PIPELINE_TASK_H_

#include "nndeploy/base/common.h"
#include "nndeploy/base/glic_stl_include.h"
#include "nndeploy/base/log.h"
#include "nndeploy/base/macro.h"
#include "nndeploy/base/object.h"
#include "nndeploy/base/status.h"
#include "nndeploy/base/string.h"
#include "nndeploy/base/value.h"
#include "nndeploy/device/buffer.h"
#include "nndeploy/device/buffer_pool.h"
#include "nndeploy/device/device.h"
#include "nndeploy/device/tensor.h"
#include "nndeploy/pipeline/packet.h"

namespace nndeploy {
namespace pipeline {

class NNDEPLOY_CC_API Task {
 public:
  Task(const std::string& name, Packet* input, Packet* output);
  Task(const std::string& name, std::vector<Packet*> inputs,
       std::vector<Packet*> outputs);

  virtual ~Task();

  std::string getName();

  virtual base::Status setParam(base::Param* param);
  virtual base::Param* getParam();

  Packet* getInput(int index = 0);
  Packet* getOutput(int index = 0);

  std::vector<Packet*> getAllInput();
  std::vector<Packet*> getAllOutput();

  bool getConstructed();
  bool getInitialized();

  bool isRunning();

  virtual base::Status init();
  virtual base::Status deinit();

  virtual base::Status reshape();

  virtual base::Status run() = 0;

 protected:
  std::string name_;
  std::shared_ptr<base::Param> param_;
  std::vector<Packet*> inputs_;
  std::vector<Packet*> outputs_;

  bool constructed_ = false;
  bool initialized_ = false;
  bool is_running_ = false;
};

using SingleIOTaskFunc = std::function<base::Status(
    Packet* input, Packet* output, base::Param* param)>;

using MultiIOTaskFunc = std::function<base::Status(std::vector<Packet*> input,
                                                   std::vector<Packet*> output,
                                                   base::Param* param)>;

}  // namespace pipeline
}  // namespace nndeploy

#endif