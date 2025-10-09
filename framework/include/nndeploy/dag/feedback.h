#ifndef _NNDEPLOY_DAG_FEEDBACK_H_
#define _NNDEPLOY_DAG_FEEDBACK_H_

#include "nndeploy/base/any.h"
#include "nndeploy/base/common.h"
#include "nndeploy/base/glic_stl_include.h"
#include "nndeploy/base/log.h"
#include "nndeploy/base/macro.h"
#include "nndeploy/base/object.h"
#include "nndeploy/base/status.h"
#include "nndeploy/base/string.h"
#include "nndeploy/dag/edge.h"
#include "nndeploy/dag/graph.h"
#include "nndeploy/dag/node.h"
#include "nndeploy/dag/util.h"
#include "nndeploy/device/buffer.h"
#include "nndeploy/device/device.h"
#include "nndeploy/device/memory_pool.h"
#include "nndeploy/device/tensor.h"

namespace nndeploy {
namespace dag {

class NNDEPLOY_CC_API Feedback : public Graph {
 public:
  Feedback(const std::string &name);
  Feedback(const std::string &name, std::vector<dag::Edge *> inputs,
           std::vector<dag::Edge *> outputs);
  ~Feedback() override;

  base::Status init() override;
  base::Status deinit() override;
  base::Status run() override;

  virtual bool condition() = 0;

 protected:
  base::Status executor() override;
};

}  // namespace dag
}  // namespace nndeploy

#endif