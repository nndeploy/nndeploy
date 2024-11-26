
#include "nndeploy/dag/edge/abstract_edge.h"

#include "nndeploy/dag/node.h"
#include "nndeploy/dag/util.h"

namespace nndeploy {
namespace dag {

AbstractEdge::AbstractEdge(base::ParallelType paralle_type)
    : parallel_type_(paralle_type) {}

AbstractEdge::~AbstractEdge() {
  producers_.clear();
  consumers_.clear();
}

base::ParallelType AbstractEdge::getParallelType() { return parallel_type_; }

std::vector<Node *> AbstractEdge::getProducers() { return producers_; }
base::Status AbstractEdge::increaseProducers(std::vector<Node *> &producers) {
  // producers_.insert(producers_.end(), producers.begin(), producers.end());
  for (auto iter : producers) {
    insertUnique(producers_, iter);
  }
  return base::kStatusCodeOk;
}
std::vector<Node *> AbstractEdge::getConsumers() { return consumers_; }
base::Status AbstractEdge::increaseConsumers(std::vector<Node *> &consumers) {
  // consumers_.insert(consumers_.end(), consumers.begin(), consumers.end());
  for (auto iter : consumers) {
    insertUnique(consumers_, iter);
  }
  return base::kStatusCodeOk;
}

bool AbstractEdge::markGraphOutput() {
  Node *node = nullptr;
  insertUnique(consumers_, node);
  this->construct();
  return true;
}

bool AbstractEdge::checkNode(const Node *node) {
  if (std::find(consumers_.begin(), consumers_.end(), node) !=
      consumers_.end()) {
    return true;
  } else {
    if (node != nullptr) {
      Node *tmp_node = const_cast<Node *>(node);
      NNDEPLOY_LOGE("This node[%s] is error.\n", tmp_node->getName().c_str());
    } else {
      NNDEPLOY_LOGE("This node is error.\n");
    }
    return false;
  }
}

std::map<base::EdgeType, std::shared_ptr<EdgeCreator>>
    &getGlobalEdgeCreatorMap() {
  static std::once_flag once;
  static std::shared_ptr<std::map<base::EdgeType, std::shared_ptr<EdgeCreator>>>
      creators;
  std::call_once(once, []() {
    creators.reset(new std::map<base::EdgeType, std::shared_ptr<EdgeCreator>>);
  });
  return *creators;
}

base::EdgeType getEdgeType(base::ParallelType type) {
  switch (type) {
    case base::kParallelTypeNone:
      return base::kEdgeTypeFixed;
    case base::kParallelTypeSequential:
      return base::kEdgeTypeFixed;
    case base::kParallelTypeTask:
      return base::kEdgeTypeFixed;
    case base::kParallelTypePipeline:
      return base::kEdgeTypePipeline;
    default:
      return base::kEdgeTypeFixed;
  }
}

AbstractEdge *createEdge(base::ParallelType type) {
  AbstractEdge *temp = nullptr;
  auto &creater_map = getGlobalEdgeCreatorMap();
  base::EdgeType edge_type = getEdgeType(type);
  if (creater_map.count(edge_type) > 0) {
    temp = creater_map[edge_type]->createEdge(type);
  }
  return temp;
}

AbstractEdge *recreateEdge(AbstractEdge *abstact_edge,
                           const base::ParallelType &paralle_type) {
  base::ParallelType cur_paralle_type = abstact_edge->getParallelType();
  AbstractEdge *new_abstact_edge = nullptr;
  if ((int)paralle_type > (int)cur_paralle_type) {
    new_abstact_edge = createEdge(paralle_type);
    if (new_abstact_edge == nullptr) {
      NNDEPLOY_LOGE("out of memory!\n");
      return nullptr;
    }
    std::vector<Node *> producers = abstact_edge->getProducers();
    new_abstact_edge->increaseProducers(producers);
    std::vector<Node *> consumers = abstact_edge->getConsumers();
    new_abstact_edge->increaseConsumers(consumers);
    delete abstact_edge;
  } else {
    new_abstact_edge = abstact_edge;
  }
  return new_abstact_edge;
}

}  // namespace dag
}  // namespace nndeploy