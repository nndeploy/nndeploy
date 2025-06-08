#include "dag/dag.h"
#include "nndeploy/base/param.h"
#include "nndeploy/dag/edge.h"
#include "nndeploy/dag/graph.h"
#include "nndeploy_api_registry.h"

namespace py = pybind11;
namespace nndeploy {
namespace dag {

NNDEPLOY_API_PYBIND11_MODULE("dag", m) {
  py::class_<ConstNode, PyConstNode<ConstNode>>(m, "ConstNode",
                                                py::dynamic_attr())
      .def(py::init<const std::string &, std::vector<Edge *>, std::vector<Edge *>>())
      .def("update_input", &ConstNode::updateInput)
      .def("init", &ConstNode::init)
      .def("deinit", &ConstNode::deinit)
      .def("run", &ConstNode::run);
}

}
}