
#include "nndeploy/dag/util.h"

namespace nndeploy {
namespace dag {

bool checkEdge(const std::vector<Edge *> &src_edges,
               const std::vector<Edge *> &dst_edges) {
  for (auto edge : src_edges) {
    bool flag = false;
    for (auto check_edge : dst_edges) {
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