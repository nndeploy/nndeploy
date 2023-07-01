
#ifndef _NNDEPLOY_SOURCE_TASK_TASK_H_
#define _NNDEPLOY_SOURCE_TASK_TASK_H_

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

class Task {
 public:
  Task(const std::string& name, Packet* input, Packet* output);
  Task(const std::string& name, std::vector<Packet*> input,
       std::vector<Packet*> output);
  virtual ~Task();

  virtual base::Status setName(const std::string& name);
  virtual std::string getName();

  virtual base::Status setParam(base::Param* param);
  virtual base::Param* getParam();

  virtual Packet* getInput(int32_t index = 0);
  virtual Packet* getOutput(int32_t index = 0);

  virtual std::vector<Packet*> getAllInput();
  virtual std::vector<Packet*> getAllOutput();

  virtual base::Status init();
  virtual base::Status deinit();

  virtual base::ShapeMap inferOuputShape();

  virtual base::Status run() = 0;

  bool getConstructed();
  bool getInitialized();

  virtual bool getVisited();
  virtual void setVisited();
  virtual void clearVisited();

  virtual bool getExecuted();
  virtual void setExecuted();
  virtual void clearExecuted();

 protected:
  std::string name_;
  std::shared_ptr<base::Param> param_;
  std::vector<Packet*> inputs_;
  std::vector<Packet*> outputs_;

  bool constructed_ = false;
  bool initialized_ = false;
  /*! \brief Used to keep track of visitation during execution and verification.
   */
  bool visited_ = false;
  /*! \brief This is used to determine if the task has executed. */
  bool executed_ = false;
};

// using TaskFunc = std::function<base::Status(Packet* input, Packet* output,
//                                             base::Param* param)>;

}  // namespace task
}  // namespace nndeploy

#endif