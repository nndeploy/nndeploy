
#include "nndeploy/dag/node.h"

#include "nndeploy/dag/edge.h"
#include "nndeploy_api_registry.h"

namespace nndeploy {

namespace dag {

NNDEPLOY_API_PYBIND11_MODULE("dag", m) { py::class_<Node>(m, "Node"); }

}  // namespace dag

}  // namespace nndeploy