
#include "nndeploy/dag/const_node.h"

namespace nndeploy {
namespace dag {

base::Status ConstNode::init() {
  return base::kStatusCodeOk;
}

base::Status ConstNode::deinit() {
  return base::kStatusCodeOk;
}

}  // namespace dag
}  // namespace nndeploy
