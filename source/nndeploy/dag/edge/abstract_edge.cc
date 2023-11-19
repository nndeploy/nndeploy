
#include "nndeploy/dag/edge/abstract_edge.h"

#include "nndeploy/dag/node.h"

namespace nndeploy {
namespace dag {

AbstractEdge::AbstractEdge(ParallelType paralle_type,
                           std::initializer_list<Node *> producers,
                           std::initializer_list<Node *> consumers)
    : paralle_type_(paralle_type),
      producers_(producers),
      consumers_(consumers) {}

AbstractEdge::~AbstractEdge() {
  producers_.clear();
  consumers_.clear();
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
    case kParallelTypeTask:
      return kEdgeTypeFixed;
    case kParallelTypePipeline:
      return kEdgeTypePipeline;
    case kParallelTypeData:
      return kEdgeTypeFixed;
    case kParallelTypeTaskPipeline:
      return kEdgeTypePipeline;
    default:
      return kEdgeTypeFixed;
  }
}

AbstractEdge *createEdge(ParallelType type,
                         std::initializer_list<Node *> producers,
                         std::initializer_list<Node *> consumers) {
  AbstractEdge *temp = nullptr;
  auto &creater_map = getGlobalEdgeCreatorMap();
  EdgeType edge_type = getEdgeType(type);
  if (creater_map.count(edge_type) > 0) {
    temp = creater_map[edge_type]->createEdge(type, producers, consumers);
  }
  return temp;
}

}  // namespace dag
}  // namespace nndeploy