
#include "nndeploy/dag/util.h"

namespace nndeploy {
namespace dag {

bool checkEdge(const std::vector<Edge*>& edges,
               const std::vector<Edge*>& check_edges) {
  for (auto edge : edges) {
    bool flag = false;
    for (auto check_edge : check_edges) {
      if (edge == check_edge) {
        flag = true;
        break;
      }
    }
    if (!flag) {
      return false;
    }
  }
  return true;
}

}  // namespace dag
}  // namespace nndeploy