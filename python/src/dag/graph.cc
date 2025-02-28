#include "nndeploy/dag/graph.h"

#include "nndeploy/base/param.h"
#include "nndeploy/dag/edge.h"
#include "nndeploy/dag/node.h"
#include "nndeploy_api_registry.h"

namespace py = pybind11;
namespace nndeploy {
namespace dag {


class PyGraph : public Graph {
 public:
  using Graph::Graph;  // 继承构造函数

  virtual base::Status init() override {
    PYBIND11_OVERRIDE(base::Status, Graph, init);
  }

  virtual base::Status deinit() override {
    PYBIND11_OVERRIDE(base::Status, Graph, deinit);
  }

  virtual base::Status run() override {
    PYBIND11_OVERRIDE(base::Status, Graph, run);
  }
};

NNDEPLOY_API_PYBIND11_MODULE("dag", m) {
  // 确保node.cc模块先被导入
  try {
    // 检查Node类是否已经在模块中注册
    if (!py::hasattr(m, "Node")) {
      // 如果Node类尚未注册，尝试导入dag模块确保node.cc已加载
      py::module::import("nndeploy._nndeploy_internal.dag");
    }
  } catch (const py::error_already_set &e) {
    // 处理导入错误，可以记录日志或继续执行
    PyErr_Clear(); // 清除异常状态
  }

  // 定义Graph类
  py::class_<Graph, PyGraph, Node, std::shared_ptr<Graph>>(m, "Graph",
                                                           py::dynamic_attr())
      .def(py::init<const std::string &>())
      .def(py::init<const std::string &, Edge *, Edge *>())
      .def(py::init<const std::string &, std::initializer_list<Edge *>,
                    std::initializer_list<Edge *>>())
      .def(py::init<const std::string &, std::vector<Edge *>,
                    std::vector<Edge *>>())
      .def("create_edge", &Graph::createEdge, py::arg("name"), py::return_value_policy::reference)
      .def("create_edge_shared_ptr", &Graph::createEdgeSharedPtr,
           py::arg("name"), py::return_value_policy::take_ownership)
      .def(
          "add_edge",
          [](Graph &g, Edge *edge, bool is_external) {
            g.addEdge(edge, is_external);
          },
          py::arg("edge"), py::arg("is_external") = true)
      .def("add_edge_shared_ptr", &Graph::addEdgeSharedPtr, py::arg("edge"))
      .def("remove_edge", &Graph::removeEdge, py::arg("edge"))
      .def("get_edge", &Graph::getEdge, py::arg("name"))
      .def("get_edge_shared_ptr", &Graph::getEdgeSharedPtr, py::arg("name"))
      .def("create_node", &Graph::createNodeByKey, py::arg("desc"))
      .def("add_node", &Graph::addNode, py::keep_alive<1, 2>(), py::arg("node"),
           py::arg("is_external") = true)
      .def("add_node_shared_ptr", &Graph::addNodeSharedPtr, py::arg("node"))
      .def("set_node_param", &Graph::setNodeParam, py::arg("node_name"),
           py::arg("param"))
      .def("get_node_param", &Graph::getNodeParam, py::arg("node_name"))
      .def("set_graph_node_share_stream", &Graph::setGraphNodeShareStream,
           py::arg("flag"))
      .def("get_graph_node_share_stream", &Graph::getGraphNodeShareStream)
      .def("update_node_io", &Graph::updateNodeIO, py::arg("node"),
           py::arg("inputs"), py::arg("outputs_name"))
      .def("init", &Graph::init)
      .def("deinit", &Graph::deinit)
      .def("run", &Graph::run)
      .def("dump", [](Graph &g) {
          g.dump(std::cout);
      });
}

}  // namespace dag
}  // namespace nndeploy