#ifndef _NNDEPLOY_DAG_LOOP_H_
#define _NNDEPLOY_DAG_LOOP_H_

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

class NNDEPLOY_CC_API Loop : public Graph {
 public:
  Loop(const std::string &name);
  Loop(const std::string &name, std::vector<dag::Edge *> inputs,
       std::vector<dag::Edge *> outputs);
  virtual ~Loop();

  virtual base::Status init();
  virtual base::Status deinit();

  virtual int loops() = 0;
  virtual base::Status run();

 protected:
  virtual base::Status executor();
};

}  // namespace dag
}  // namespace nndeploy

#endif /* _NNDEPLOY_DAG_LOOP_H_ */