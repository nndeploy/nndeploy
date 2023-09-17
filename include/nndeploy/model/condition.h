#ifndef _NNDEPLOY_MODEL_CONDITION_H_
#define _NNDEPLOY_MODEL_CONDITION_H_

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
#include "nndeploy/model/infer.h"
#include "nndeploy/model/packet.h"
#include "nndeploy/model/task.h"

namespace nndeploy {
namespace model {

class NNDEPLOY_CC_API Condition : public Task {
 public:
  Condition(const std::string& name, Packet* input, Packet* output);
  Condition(const std::string& name, std::vector<Packet*> inputs,
            std::vector<Packet*> outputs);
  virtual ~Condition();

  template <typename T,
            typename std::enable_if<std::is_base_of<Task, T>{}, int>::type = 0>
  Task* createTask(const std::string& name, Packet* input, Packet* output) {
    NNDEPLOY_CHECK_PARAM_NULL_RET_NULL(input, "input is null!");
    NNDEPLOY_CHECK_PARAM_NULL_RET_NULL(output, "output is null!");
    bool flag = check({input}, inputs_);
    if (!flag) {
      NNDEPLOY_LOGE("input is not in condition inputs!\n");
      return nullptr;
    }
    flag = check({output}, outputs_);
    if (!flag) {
      NNDEPLOY_LOGE("output is not in condition outputs!\n");
      return nullptr;
    }
    Task* task = dynamic_cast<Task*>(new T(name, input, output));
    condition_task_.push_back(task);
    return task;
  }
  template <typename T,
            typename std::enable_if<std::is_base_of<Task, T>{}, int>::type = 0>
  Task* createTask(const std::string& name, std::vector<Packet*> inputs,
                   std::vector<Packet*> outputs) {
    if (inputs.empty() || outputs.empty()) {
      NNDEPLOY_LOGE("inputs or outputs is empty!\n");
      return nullptr;
    }
    bool flag = check(inputs, inputs_);
    if (!flag) {
      NNDEPLOY_LOGE("inputs is not in condition inputs!\n");
      return nullptr;
    }
    flag = check(outputs, outputs_);
    if (!flag) {
      NNDEPLOY_LOGE("outputs is not in condition outputs!\n");
      return nullptr;
    }
    Task* task = dynamic_cast<Task*>(new T(name, inputs, outputs));
    condition_task_.push_back(task);
    return task;
  }
  template <typename T,
            typename std::enable_if<std::is_base_of<Task, T>{}, int>::type = 0>
  Task* createInfer(const std::string& name, base::InferenceType type,
                    Packet* input, Packet* output) {
    NNDEPLOY_CHECK_PARAM_NULL_RET_NULL(input, "input is null!");
    NNDEPLOY_CHECK_PARAM_NULL_RET_NULL(output, "output is null!");
    bool flag = check({input}, inputs_);
    if (!flag) {
      NNDEPLOY_LOGE("input is not in condition inputs!\n");
      return nullptr;
    }
    flag = check({output}, outputs_);
    if (!flag) {
      NNDEPLOY_LOGE("output is not in condition outputs!\n");
      return nullptr;
    }
    Task* task = dynamic_cast<Task*>(new T(name, type, input, output));
    condition_task_.push_back(task);
    return task;
  }

  base::Status setTaskParam(const std::string& task_name, base::Param* param);
  base::Param* getTaskParam(const std::string& task_name);

  virtual base::Status init();
  virtual base::Status deinit();

  virtual int choose() = 0;

  virtual base::Status reshape();

  virtual base::Status run();

 private:
  bool check(const std::vector<Packet*>& packets,
             const std::vector<Packet*>& condition_packets);
  Task* findTask(const std::string& name);

 protected:
  std::vector<Task*> condition_task_;
};

}  // namespace model
}  // namespace nndeploy

#endif /* _NNDEPLOY_MODEL_CONDITION_H_ */