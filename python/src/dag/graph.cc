#include "nndeploy/dag/graph.h"

#include "dag/dag.h"
#include "nndeploy/base/param.h"
#include "nndeploy/dag/edge.h"
#include "nndeploy/dag/node.h"
#include "nndeploy_api_registry.h"

namespace py = pybind11;
namespace nndeploy {
namespace dag {

NNDEPLOY_API_PYBIND11_MODULE("dag", m) {
  // 定义Graph类
  py::class_<Graph, Node, PyGraph<Graph>>(m, "Graph", py::dynamic_attr())
      .def(py::init<const std::string &>())
      .def(py::init<const std::string &, std::vector<Edge *>,
                    std::vector<Edge *>>())
      .def("set_edge_queue_max_size", &Graph::setEdgeQueueMaxSize,
           py::arg("queue_max_size"))
      .def("get_edge_queue_max_size", &Graph::getEdgeQueueMaxSize)
      .def("set_input", &Graph::setInput, py::arg("input"),
           py::arg("index") = -1)
      .def("set_output", &Graph::setOutput, py::arg("output"),
           py::arg("index") = -1)
      .def("set_inputs", &Graph::setInputs, py::arg("inputs"))
      .def("set_outputs", &Graph::setOutputs, py::arg("outputs"))
      //  .def("set_input_shared_ptr", &Graph::setInputSharedPtr,
      //  py::arg("input"),
      //       py::arg("index") = -1)
      //  .def("set_output_shared_ptr", &Graph::setOutputSharedPtr,
      //       py::arg("output"), py::arg("index") = -1)
      //  .def("set_inputs_shared_ptr", &Graph::setInputsSharedPtr,
      //       py::arg("inputs"))
      //  .def("set_outputs_shared_ptr", &Graph::setOutputsSharedPtr,
      //       py::arg("outputs"))
      .def("create_edge", &Graph::createEdge, py::arg("name"),
           py::return_value_policy::reference)
      .def(
          "add_edge",
          [](Graph &g, Edge *edge) {
            bool is_external = true;
            g.addEdge(edge, is_external);
          },
          py::arg("edge"), py::keep_alive<1, 2>())
      .def("update_edge", &Graph::updteEdge, py::arg("edge_wrapper"),
           py::arg("edge"), py::arg("is_external") = true)
      .def("get_edge", &Graph::getEdge, py::arg("name"),
           py::return_value_policy::reference)
      .def("create_node",
           py::overload_cast<const std::string &, const std::string &>(
               &Graph::createNode4Py),
           py::arg("key"), py::arg("name"),
           py::return_value_policy::take_ownership)
      .def("create_node",
           py::overload_cast<const NodeDesc &>(&Graph::createNode4Py),
           py::arg("desc"), py::return_value_policy::take_ownership)
      .def(
          "set_node_desc",
          [](Graph &g, Node *node, const NodeDesc &desc) {
            //   NNDEPLOY_LOGE("set_node_desc[%s, %p] success!\n",
            //                 node->getName().c_str(), node);
            return g.setNodeDesc(node, desc);
          },
          py::arg("node"), py::arg("desc"))
      .def(
          "add_node",
          [](Graph &g, Node *node) {
            bool is_external = true;
            g.addNode(node, is_external);
          },
          py::keep_alive<1, 2>(), py::arg("node"))
      .def("get_node", &Graph::getNode, py::arg("name"),
           py::return_value_policy::reference)
      //  .def("get_node_shared_ptr", &Graph::getNodeSharedPtr, py::arg("name"))
      .def("get_node_by_key", &Graph::getNodeByKey, py::arg("key"),
           py::return_value_policy::reference)
      .def("get_nodes_by_key", &Graph::getNodesByKey, py::arg("key"),
           py::return_value_policy::reference)
      .def("set_node_param", &Graph::setNodeParamSharedPtr,
           py::arg("node_name"), py::arg("param"))
      .def("get_node_param", &Graph::getNodeParamSharedPtr,
           py::arg("node_name"))
      .def("set_graph_node_share_stream", &Graph::setGraphNodeShareStream,
           py::arg("flag"))
      .def("get_graph_node_share_stream", &Graph::getGraphNodeShareStream)
      .def("update_node_io", &Graph::updateNodeIO, py::arg("node"),
           py::arg("inputs"), py::arg("outputs"))
      .def("mark_input_edge", &Graph::markInputEdge, py::arg("inputs"))
      .def("mark_output_edge", &Graph::markOutputEdge, py::arg("outputs"))
      .def("default_param", &Graph::defaultParam)
      .def("init", &Graph::init)
      .def("deinit", &Graph::deinit)
      .def("run", &Graph::run)
      .def("forward", &Graph::forward, py::arg("inputs"),
           py::keep_alive<1, 2>(), py::return_value_policy::reference)
      .def("__call__", &Graph::operator(), py::arg("inputs"),
           py::keep_alive<1, 2>(), py::return_value_policy::reference)
      .def("dump", [](Graph &g) { g.dump(std::cout); })
      .def("set_trace_flag", &Graph::setTraceFlag, py::arg("flag"))
      // 绑定Graph类的trace方法到Python
      // 参数:
      //   inputs: 输入的边列表
      // 返回:
      //   返回追踪后的边列表
      // 说明:
      //   1. py::keep_alive<1,2>() 确保Graph对象在inputs存在期间不会被销毁
      //   2. py::return_value_policy::reference 返回引用,由Python管理内存
      //   3. 该方法用于追踪图的执行流程,帮助调试和性能分析
      .def("trace", &Graph::trace, py::arg("inputs"), py::keep_alive<1, 2>(),
           py::return_value_policy::reference)
      .def("get_edge_wrapper",
           py::overload_cast<Edge *>(&Graph::getEdgeWrapper), py::arg("edge"),
           py::return_value_policy::reference)
      .def("get_edge_wrapper",
           py::overload_cast<const std::string &>(&Graph::getEdgeWrapper),
           py::arg("name"), py::return_value_policy::reference)
      .def("get_node_wrapper",
           py::overload_cast<Node *>(&Graph::getNodeWrapper), py::arg("node"),
           py::return_value_policy::reference)
      .def("get_node_wrapper",
           py::overload_cast<const std::string &>(&Graph::getNodeWrapper),
           py::arg("name"), py::return_value_policy::reference)
      .def("serialize",
           py::overload_cast<rapidjson::Value &,
                             rapidjson::Document::AllocatorType &>(
               &Graph::serialize),
           py::arg("json"), py::arg("allocator"))
      .def("serialize", py::overload_cast<>(&Graph::serialize))
      .def("deserialize",
           py::overload_cast<rapidjson::Value &>(&Graph::deserialize),
           py::arg("json"))
      .def("deserialize",
           py::overload_cast<const std::string &>(&Graph::deserialize),
           py::arg("json_str"));

  m.def("serialize", py::overload_cast<Graph *>(&serialize), py::arg("graph"));
  m.def("save_file", &saveFile, py::arg("graph"), py::arg("path"));
  m.def("deserialize", py::overload_cast<const std::string &>(&deserialize),
        py::arg("json_str"), py::return_value_policy::take_ownership);
  m.def("load_file", &loadFile, py::arg("path"),
        py::return_value_policy::take_ownership);
}

}  // namespace dag
}  // namespace nndeploy