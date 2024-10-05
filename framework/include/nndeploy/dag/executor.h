
#ifndef _NNDEPLOY_DAG_EXECUTOR_H_
#define _NNDEPLOY_DAG_EXECUTOR_H_

#include "nndeploy/base/common.h"
#include "nndeploy/base/glic_stl_include.h"
#include "nndeploy/base/log.h"
#include "nndeploy/base/macro.h"
#include "nndeploy/base/object.h"
#include "nndeploy/base/status.h"
#include "nndeploy/base/string.h"
#include "nndeploy/base/value.h"
#include "nndeploy/dag/edge.h"
#include "nndeploy/dag/node.h"
#include "nndeploy/dag/util.h"

namespace nndeploy {
namespace dag {

class NNDEPLOY_CC_API Executor : public base::NonCopyable {
 public:
  Executor() {};
  virtual ~Executor() {};

  virtual base::Status init(std::vector<EdgeWrapper *> &edge_repository,
                            std::vector<NodeWrapper *> &node_repository) = 0;
  virtual base::Status deinit() = 0;

  virtual base::Status run() = 0;
};

}  // namespace dag
}  // namespace nndeploy

#endif