
#include "nndeploy/dag/edge/abstract_edge.h"

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

std::map<ParallelType, std::shared_ptr<EdgeCreator>>
    &getGlobalEdgeCreatorMap() {
  static std::once_flag once;
  static std::shared_ptr<std::map<ParallelType, std::shared_ptr<EdgeCreator>>>
      creators;
  std::call_once(once, []() {
    creators.reset(new std::map<ParallelType, std::shared_ptr<EdgeCreator>>);
  });
  return *creators;
}

AbstractEdge *createEdge(ParallelType type,
                         std::initializer_list<Node *> producers,
                         std::initializer_list<Node *> consumers) {
  AbstractEdge *temp = nullptr;
  auto &creater_map = getGlobalEdgeCreatorMap();
  if (creater_map.count(type) > 0) {
    temp = creater_map[type]->createEdge(type, producers, consumers);
  }
  return temp;
}

}  // namespace dag
}  // namespace nndeploy