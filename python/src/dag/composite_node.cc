#include "dag/dag.h"
#include "nndeploy/base/param.h"
#include "nndeploy/dag/edge.h"
#include "nndeploy/dag/graph.h"
#include "nndeploy_api_registry.h"

namespace py = pybind11;
namespace nndeploy {
namespace dag {

NNDEPLOY_API_PYBIND11_MODULE("dag", m) {
  py::class_<CompositeNode, PyCompositeNode<CompositeNode>>(m, "CompositeNode",
                                                            py::dynamic_attr())
      .def(py::init<const std::string &, std::vector<Edge *>,
                    std::vector<Edge *>>())
      .def("init", &CompositeNode::init)
      .def("deinit", &CompositeNode::deinit)
      .def("run", &CompositeNode::run)
      .def(
          "create_node",
          [](CompositeNode &self, const NodeDesc &desc) {
            return self.createNode(desc);
          },
          py::arg("desc"), py::return_value_policy::reference)
      .def("find_edge_by_name", &CompositeNode::findEdgeByName,
           py::arg("edges"), py::arg("name"),
           py::return_value_policy::reference)
      .def("get_edge", &CompositeNode::getEdge, py::arg("name"),
           py::return_value_policy::reference)
      .def("create_edge", &CompositeNode::createEdge, py::arg("name"),
           py::return_value_policy::reference)
      .def("sort_dfs", &CompositeNode::sortDFS);
}

}  // namespace dag
}  // namespace nndeploy