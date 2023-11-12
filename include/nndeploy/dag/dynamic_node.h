#ifndef _NNDEPLOY_DAG_DYNAMIC_NODE_H_
#define _NNDEPLOY_DAG_DYNAMIC_NODE_H_
#include "nndeploy/dag/dynamic_edge.h"
namespace nndeploy {
namespace dag {
class NNDEPLOY_CC_API DynamicNode {
 public:
  DynamicNode() {}
  ~DynamicNode() {}
  std::vector<DataPacket *> getInput();
  base::Status setOutput(std::vector<DataPacket *> result);
  base::Status run();

 private:
  std::vector<DynamicEdge *> inputs_ = {};
  std::vector<DynamicEdge *> outputs_ = {};
  int idx_ = 0;
}

}  // namespace dag
}  // namespace nndeploy
#endif