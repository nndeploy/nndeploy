
#ifndef _NNDEPLOY_DAG_RUNNING_CONDITION__H_
#define _NNDEPLOY_DAG_RUNNING_CONDITION__H_

#include "nndeploy/base/any.h"
#include "nndeploy/base/common.h"
#include "nndeploy/base/glic_stl_include.h"
#include "nndeploy/base/log.h"
#include "nndeploy/base/macro.h"
#include "nndeploy/base/object.h"
#include "nndeploy/base/status.h"
#include "nndeploy/base/string.h"
#include "nndeploy/dag/condition.h"
#include "nndeploy/dag/edge.h"
#include "nndeploy/dag/node.h"
#include "nndeploy/dag/util.h"

namespace nndeploy {
namespace dag {

class NNDEPLOY_CC_API RunningCondition : public Condition {
 public:
  RunningCondition(const std::string &name);
  // RunningCondition(const std::string &name, Edge *input, Edge *output);
  RunningCondition(const std::string &name,
                   std::initializer_list<Edge *> inputs,
                   std::initializer_list<Edge *> outputs);
  RunningCondition(const std::string &name, std::vector<Edge *> inputs,
                   std::vector<Edge *> outputs);
  virtual ~RunningCondition();

  virtual int choose();

 protected:
  int next_i_ = 0;
};

}  // namespace dag
}  // namespace nndeploy

#endif /* _NNDEPLOY_DAG_RUNNING_CONDITION__H_ */
