#ifndef _NNDEPLOY_DAG_UTIL_H_
#define _NNDEPLOY_DAG_UTIL_H_

#include "nndeploy/base/common.h"
#include "nndeploy/base/glic_stl_include.h"
#include "nndeploy/base/log.h"
#include "nndeploy/base/macro.h"
#include "nndeploy/base/object.h"
#include "nndeploy/base/status.h"
#include "nndeploy/base/string.h"
#include "nndeploy/dag/edge.h"
#include "nndeploy/dag/node.h"

namespace nndeploy {
namespace dag {

class NNDEPLOY_CC_API NodeWrapper {
 public:
  bool is_external_;
  Node *node_;
  std::string name_;
  std::vector<NodeWrapper *> predecessors_;
  std::vector<NodeWrapper *> successors_;
  base::NodeColorType color_ = base::kNodeColorWhite;
};

class NNDEPLOY_CC_API EdgeWrapper {
 public:
  bool is_external_;
  Edge *edge_;
  std::string name_;
  std::vector<NodeWrapper *> producers_;
  std::vector<NodeWrapper *> consumers_;
};

extern NNDEPLOY_CC_API Edge *getEdge(std::vector<EdgeWrapper *> &edge_repository,
                              const std::string &edge_name);
extern NNDEPLOY_CC_API EdgeWrapper *findEdgeWrapper(
    std::vector<EdgeWrapper *> &edge_repository, const std::string &edge_name);
extern NNDEPLOY_CC_API EdgeWrapper *findEdgeWrapper(
    std::vector<EdgeWrapper *> &edge_repository, Edge *edge);
extern NNDEPLOY_CC_API std::vector<EdgeWrapper *> findStartEdges(
    std::vector<EdgeWrapper *> &edge_repository);
extern NNDEPLOY_CC_API std::vector<EdgeWrapper *> findEndEdges(
    std::vector<EdgeWrapper *> &edge_repository);

extern NNDEPLOY_CC_API Node *getNode(std::vector<NodeWrapper *> &node_repository,
                              const std::string &node_name);
extern NNDEPLOY_CC_API NodeWrapper *findNodeWrapper(
    std::vector<NodeWrapper *> &node_repository, const std::string &node_name);
extern NNDEPLOY_CC_API NodeWrapper *findNodeWrapper(
    std::vector<NodeWrapper *> &node_repository, Node *node);
extern NNDEPLOY_CC_API std::vector<NodeWrapper *> findStartNodes(
    std::vector<NodeWrapper *> &node_repository);
extern NNDEPLOY_CC_API std::vector<NodeWrapper *> findEndNodes(
    std::vector<NodeWrapper *> &node_repository);

extern NNDEPLOY_CC_API base::Status setColor(
    std::vector<NodeWrapper *> &node_repository, base::NodeColorType color);

extern NNDEPLOY_CC_API base::Status dumpDag(std::vector<EdgeWrapper *> &edge_repository,
                     std::vector<NodeWrapper *> &node_repository,
                     std::vector<Edge *> &graph_inputs,
                     std::vector<Edge *> &graph_outputs,
                     const std::string &name, std::ostream &oss);

extern NNDEPLOY_CC_API std::vector<NodeWrapper *> checkUnuseNode(
    std::vector<NodeWrapper *> &node_repository);
extern NNDEPLOY_CC_API std::vector<EdgeWrapper *> checkUnuseEdge(
    std::vector<NodeWrapper *> &node_repository,
    std::vector<EdgeWrapper *> &edge_repository);

extern NNDEPLOY_CC_API base::Status topoSortBFS(std::vector<NodeWrapper *> &node_repository,
                         std::vector<NodeWrapper *> &topo_sort_node);

extern NNDEPLOY_CC_API base::Status topoSortDFS(std::vector<NodeWrapper *> &node_repository,
                         std::vector<NodeWrapper *> &topo_sort_node);

extern NNDEPLOY_CC_API base::Status topoSort(std::vector<NodeWrapper *> &node_repository,
                      base::TopoSortType topo_sort_type,
                      std::vector<NodeWrapper *> &topo_sort_node);

extern NNDEPLOY_CC_API bool checkEdge(const std::vector<Edge *> &src_edges,
               const std::vector<Edge *> &dst_edges);

/**
 * @brief 对vector插入不在vector中的元素，即类似集合的作用
 * @tparam T
 * @param  vec              My Param doc
 * @param  val              My Param doc
 */
template <typename T>
void insertUnique(std::vector<T> &vec, const T &val) {
  if (std::find(vec.begin(), vec.end(), val) == vec.end()) {
    vec.emplace_back(val);
  }
}

extern NNDEPLOY_CC_API void findConsumerNode(EdgeWrapper *edge_wrapper,
                      std::vector<Node *> &consumers);

extern NNDEPLOY_CC_API void findProducerNode(EdgeWrapper *edge_wrapper,
                      std::vector<Node *> &producers);

// base::Status getValueFromParamVec(const std::vector<base::Param *>
// &param_vec,
// const std::string &key, base::Value
// &value);

}  // namespace dag
}  // namespace nndeploy

#endif /* _NNDEPLOY_DAG_UTIL_H_ */
