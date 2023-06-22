
#ifndef _NNDEPLOY_SOURCE_TASK_EXECUTION_H_
#define _NNDEPLOY_SOURCE_TASK_EXECUTION_H_

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
#include "nndeploy/source/task/packet.h"

namespace nndeploy {
namespace task {

class Execution {
 public:
  Execution(const std::string& name = "");
  virtual ~Execution();

  virtual base::Param* getParam();

  virtual base::Status init();
  virtual base::Status deinit();

  virtual Packet* getInput();
  virtual Packet* getOutput();

  virtual base::Status setInput(Packet& input);
  virtual base::Status setOutput(Packet& output);

  virtual base::ShapeMap inferShape(
      base::ShapeMap min_shape = base::ShapeMap(),
      base::ShapeMap opt_shape = base::ShapeMap(),
      base::ShapeMap max_shape = base::ShapeMap());

  virtual base::Status run() = 0;

 protected:
  std::string name_;
  std::shared_ptr<base::Param> param_;
  Packet* input_ = nullptr;
  Packet* output_ = nullptr;
};

using ExecutionFunc = std::function<base::Status(
    const Packet& input, Packet& output, std::shared_ptr<base::Param> param)>;

}  // namespace task
}  // namespace nndeploy

#endif