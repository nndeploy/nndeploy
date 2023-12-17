
#ifndef _NNDEPLOY_DAG_GRAPH_EXECUTOR_H_
#define _NNDEPLOY_DAG_GRAPH_EXECUTOR_H_

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
#include "nndeploy/device/buffer.h"
#include "nndeploy/device/buffer_pool.h"
#include "nndeploy/device/device.h"
#include "nndeploy/device/tensor.h"
#include "nndeploy/thread_pool/safe_ws_queue.h"
#include "nndeploy/thread_pool/thread_pool.h"

namespace nndeploy {
namespace dag {

enum NodeColorType : int {
  kNodeColorWhite = 0x0000,
  kNodeColorGray,
  kNodeColorBlack
};

enum TopoSortType : int { kTopoSortTypeBFS = 0x0000, kTopoSortTypeDFS };

class NodeWrapper {
 public:
  bool is_external_;
  Node* node_;
  std::string name_;
  std::vector<NodeWrapper*> predecessors_;
  std::vector<NodeWrapper*> successors_;
  NodeColorType color_ = kNodeColorWhite;
};

class EdgeWrapper {
 public:
  bool is_external_;
  Edge* edge_;
  std::string name_;
  std::vector<NodeWrapper*> producers_;
  std::vector<NodeWrapper*> consumers_;
};

class Executor : public base::NonCopyable {
 public:
  Executor(){};
  virtual ~Executor(){};

  virtual base::Status init(std::vector<EdgeWrapper*>& edge_repository,
                            std::vector<NodeWrapper*>& node_repository) = 0;
  virtual base::Status deinit() = 0;

  virtual base::Status run() = 0;
};

Edge* getEdge(std::vector<EdgeWrapper*>& edge_repository,
              const std::string& edge_name);
EdgeWrapper* findEdgeWrapper(std::vector<EdgeWrapper*>& edge_repository,
                             const std::string& edge_name);
EdgeWrapper* findEdgeWrapper(std::vector<EdgeWrapper*>& edge_repository,
                             Edge* edge);
std::vector<EdgeWrapper*> findStartEdges(
    std::vector<EdgeWrapper*>& edge_repository);
std::vector<EdgeWrapper*> findEndEdges(
    std::vector<EdgeWrapper*>& edge_repository);

Node* getNode(std::vector<NodeWrapper*>& node_repository,
              const std::string& node_name);
NodeWrapper* findNodeWrapper(std::vector<NodeWrapper*>& node_repository,
                             const std::string& node_name);
NodeWrapper* findNodeWrapper(std::vector<NodeWrapper*>& node_repository,
                             Node* node);
std::vector<NodeWrapper*> findStartNodes(
    std::vector<NodeWrapper*>& node_repository);
std::vector<NodeWrapper*> findEndNodes(
    std::vector<NodeWrapper*>& node_repository);

base::Status dumpDag(std::vector<NodeWrapper*>& node_repository,
                     const std::string& name = "",
                     std::ostream& oss = std::cout);

base::Status topoSortBFS(std::vector<NodeWrapper*>& node_repository,
                         std::vector<NodeWrapper*>& topo_sort_node);

base::Status topoSortDFS(std::vector<NodeWrapper*>& node_repository,
                         std::vector<NodeWrapper*>& topo_sort_node);

base::Status topoSort(std::vector<NodeWrapper*>& node_repository,
                      TopoSortType topo_sort_type,
                      std::vector<NodeWrapper*>& topo_sort_node);

base::Status setColor(std::vector<NodeWrapper*>& node_repository,
                      NodeColorType color);

base::Status commitTask(
    NodeWrapper* node_wrapper, thread_pool::ThreadPool* thread_pool,
    thread_pool::SafeWSQueue<std::future<base::Status>>& end_tasks);

}  // namespace dag
}  // namespace nndeploy

#endif