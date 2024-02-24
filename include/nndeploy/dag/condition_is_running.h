
#ifndef _NNDEPLOY_DAG_CONDITION_IS_RUNNING_H_
#define _NNDEPLOY_DAG_CONDITION_IS_RUNNING_H_

#include "nndeploy/base/common.h"
#include "nndeploy/base/glic_stl_include.h"
#include "nndeploy/base/log.h"
#include "nndeploy/base/macro.h"
#include "nndeploy/base/object.h"
#include "nndeploy/base/status.h"
#include "nndeploy/base/string.h"
#include "nndeploy/base/value.h"
#include "nndeploy/dag/condition.h"
#include "nndeploy/dag/edge.h"
#include "nndeploy/dag/node.h"
#include "nndeploy/dag/util.h"
#include "nndeploy/device/buffer.h"
#include "nndeploy/device/buffer_pool.h"
#include "nndeploy/device/device.h"
#include "nndeploy/device/tensor.h"
#include "nndeploy/thread_pool/thread_pool.h"

namespace nndeploy {
namespace dag {

class NNDEPLOY_CC_API ConditionIsRunning : public Condition {
 public:
  ConditionIsRunning(const std::string &name, Edge *input, Edge *output);
  ConditionIsRunning(const std::string &name,
                     std::initializer_list<Edge *> inputs,
                     std::initializer_list<Edge *> outputs);
  virtual ~ConditionIsRunning();

  virtual base::Status init();
  virtual base::Status deinit();

  virtual int choose();

  virtual base::Status run();

 protected:
  thread_pool::ThreadPool *thread_pool_ = nullptr;
  int all_task_count_ = 0;
};

}  // namespace dag
}  // namespace nndeploy

#endif /* _NNDEPLOY_DAG_CONDITION_IS_RUNNING_H_ */
