#include "nndeploy/dag/graph.h"

#include "dag/dag.h"
#include "nndeploy/base/param.h"
#include "nndeploy/dag/edge.h"
#include "nndeploy/dag/node.h"
#include "nndeploy_api_registry.h"

namespace py = pybind11;
namespace nndeploy {
namespace dag {

// class PyGraph : public Graph {
//  public:
//   using Graph::Graph;  // 继承构造函数

//   virtual base::Status init() override {
//     PYBIND11_OVERRIDE_NAME(base::Status, Graph, "init", init);
//   }

//   virtual base::Status deinit() override {
//     PYBIND11_OVERRIDE_NAME(base::Status, Graph, "deinit", deinit);
//   }

//   virtual base::Status run() override {
//     PYBIND11_OVERRIDE_NAME(base::Status, Graph, "run", run);
//   }
// };

NNDEPLOY_API_PYBIND11_MODULE("dag", m) {
  // 定义Graph类
  py::class_<Graph, Node, PyGraph<Graph>>(m, "Graph", py::dynamic_attr())
      .def(py::init<const std::string &>())
      .def(py::init<const std::string &, Edge *, Edge *>())
      .def(py::init<const std::string &, std::initializer_list<Edge *>,
                    std::initializer_list<Edge *>>())
      .def(py::init<const std::string &, std::vector<Edge *>,
                    std::vector<Edge *>>())
      .def("create_edge", &Graph::createEdge, py::arg("name"),
           py::return_value_policy::reference)
      .def(
          "add_edge",
          [](Graph &g, Edge *edge) {
            bool is_external = true;
            g.addEdge(edge, is_external);
          },
          py::arg("edge"), py::keep_alive<1, 2>())
      .def("remove_edge", &Graph::removeEdge, py::arg("edge"))
      .def("get_edge", &Graph::getEdge, py::arg("name"),
           py::return_value_policy::reference)
      .def("create_node", &Graph::createNodeByKey, py::arg("desc"),
           py::return_value_policy::reference)
      .def(
          "add_node",
          [](Graph &g, Node *node) {
            bool is_external = true;
            g.addNode(node, is_external);
          },
          py::keep_alive<1, 2>(), py::arg("node"))
      .def("set_node_param", &Graph::setNodeParamSharedPtr,
           py::arg("node_name"), py::arg("param"))
      .def("get_node_param", &Graph::getNodeParamSharedPtr,
           py::arg("node_name"))
      .def("set_graph_node_share_stream", &Graph::setGraphNodeShareStream,
           py::arg("flag"))
      .def("get_graph_node_share_stream", &Graph::getGraphNodeShareStream)
      .def("update_node_io",
           py::overload_cast<Node *, std::vector<std::shared_ptr<Edge>>,
                             std::vector<std::string>>(&Graph::updateNodeIO),
           py::arg("node"), py::arg("inputs"), py::arg("outputs_name"))
      .def(
          "update_node_io",
          py::overload_cast<Node *, std::vector<Edge *>,
                            std::vector<std::string>,
                            std::shared_ptr<base::Param>>(&Graph::updateNodeIO),
          py::arg("node"), py::arg("inputs"), py::arg("outputs_name"),
          py::arg("param"))
      .def("init", &Graph::init)
      .def("deinit", &Graph::deinit)
      .def("run", &Graph::run)
      .def("dump", [](Graph &g) { g.dump(std::cout); });
}

}  // namespace dag
}  // namespace nndeploy