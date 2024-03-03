
#include "nndeploy/dag/edge/abstract_edge.h"

#include "nndeploy/dag/node.h"
#include "nndeploy/dag/util.h"

namespace nndeploy {
namespace dag {

AbstractEdge::AbstractEdge(ParallelType paralle_type)
    : parallel_type_(paralle_type) {}

AbstractEdge::~AbstractEdge() {
  producers_.clear();
  consumers_.clear();
}

ParallelType AbstractEdge::getParallelType() { return parallel_type_; }

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

std::map<EdgeType, std::shared_ptr<EdgeCreator>> &getGlobalEdgeCreatorMap() {
  static std::once_flag once;
  static std::shared_ptr<std::map<EdgeType, std::shared_ptr<EdgeCreator>>>
      creators;
  std::call_once(once, []() {
    creators.reset(new std::map<EdgeType, std::shared_ptr<EdgeCreator>>);
  });
  return *creators;
}

EdgeType getEdgeType(ParallelType type) {
  switch (type) {
    case kParallelTypeNone:
      return kEdgeTypeFixed;
    case kParallelTypeSequential:
      return kEdgeTypeFixed;
    case kParallelTypeTask:
      return kEdgeTypeFixed;
    case kParallelTypePipeline:
      return kEdgeTypePipeline;
    default:
      return kEdgeTypeFixed;
  }
}

AbstractEdge *createEdge(ParallelType type) {
  AbstractEdge *temp = nullptr;
  auto &creater_map = getGlobalEdgeCreatorMap();
  EdgeType edge_type = getEdgeType(type);
  if (creater_map.count(edge_type) > 0) {
    temp = creater_map[edge_type]->createEdge(type);
  }
  return temp;
}

}  // namespace dag
}  // namespace nndeploy