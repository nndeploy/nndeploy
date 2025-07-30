// #include "dag/dag.h"
// #include "nndeploy/base/param.h"
// #include "nndeploy/dag/edge.h"
// #include "nndeploy/dag/graph.h"
// #include "nndeploy_api_registry.h"

// namespace py = pybind11;
// namespace nndeploy {
// namespace dag {

// NNDEPLOY_API_PYBIND11_MODULE("dag", m) {
//   py::class_<CompositeNode, Node, PyCompositeNode<CompositeNode>>(
//       m, "CompositeNode", py::dynamic_attr())
//       .def(py::init<const std::string &>())
//       .def(py::init<const std::string &, std::vector<Edge *>,
//                     std::vector<Edge *>>())
//       .def("set_input", &CompositeNode::setInput, py::arg("input"),
//            py::arg("index") = -1)
//       .def("set_output", &CompositeNode::setOutput, py::arg("output"),
//            py::arg("index") = -1)
//       .def("set_inputs", &CompositeNode::setInputs, py::arg("inputs"))
//       .def("set_outputs", &CompositeNode::setOutputs, py::arg("outputs"))
//       .def("create_edge", &CompositeNode::createEdge, py::arg("name"),
//            py::return_value_policy::reference)
//       .def("add_edge", &CompositeNode::addEdge, py::arg("edge"),
//            py::arg("is_external") = true)
//       .def("get_edge", &CompositeNode::getEdge, py::arg("name"),
//            py::return_value_policy::reference)
//       .def("update_edge", &CompositeNode::updteEdge, py::arg("edge_wrapper"),
//            py::arg("edge"), py::arg("is_external") = true)
//       .def("create_node",
//            py::overload_cast<const NodeDesc &>(&CompositeNode::createNode4Py),
//            py::arg("desc"), py::return_value_policy::take_ownership)
//       .def("set_node_desc", &CompositeNode::setNodeDesc, py::arg("node"),
//            py::arg("desc"))
//       .def(
//           "add_node",
//           [](CompositeNode &c, Node *node) {
//             bool is_external = true;
//             c.addNode(node, is_external);
//           },
//           py::arg("node"), py::keep_alive<1, 2>())
//       .def("update_node_io", &CompositeNode::updateNodeIO, py::arg("node"),
//            py::arg("inputs"), py::arg("outputs"))
//       .def("mark_input_edge", &CompositeNode::markInputEdge, py::arg("inputs"))
//       .def("mark_output_edge", &CompositeNode::markOutputEdge,
//            py::arg("outputs"))
//       .def("get_node", &CompositeNode::getNode, py::arg("name"),
//            py::return_value_policy::reference)
//       .def("get_node_by_key", &CompositeNode::getNodeByKey, py::arg("key"),
//            py::return_value_policy::reference)
//       .def("get_nodes_by_key", &CompositeNode::getNodesByKey, py::arg("key"),
//            py::return_value_policy::reference)
//       .def("set_node_param", &CompositeNode::setNodeParamSharedPtr,
//            py::arg("node_name"), py::arg("param"))
//       .def("get_node_param", &CompositeNode::getNodeParamSharedPtr,
//            py::arg("node_name"), py::return_value_policy::reference)
//       .def("default_param", &CompositeNode::defaultParam)
//       .def("init", &CompositeNode::init)
//       .def("deinit", &CompositeNode::deinit)
//       .def("run", &CompositeNode::run)
//       .def("get_edge_wrapper",
//            py::overload_cast<Edge *>(&CompositeNode::getEdgeWrapper),
//            py::arg("edge"), py::return_value_policy::reference)
//       .def("get_edge_wrapper",
//            py::overload_cast<const std::string &>(
//                &CompositeNode::getEdgeWrapper),
//            py::arg("name"), py::return_value_policy::reference)
//       .def("get_node_wrapper",
//            py::overload_cast<Node *>(&CompositeNode::getNodeWrapper),
//            py::arg("node"), py::return_value_policy::reference)
//       .def("get_node_wrapper",
//            py::overload_cast<const std::string &>(
//                &CompositeNode::getNodeWrapper),
//            py::arg("name"), py::return_value_policy::reference)
//       .def("serialize",
//            py::overload_cast<rapidjson::Value &,
//                              rapidjson::Document::AllocatorType &>(
//                &CompositeNode::serialize),
//            py::arg("json"), py::arg("allocator"))
//       .def("serialize", py::overload_cast<>(&CompositeNode::serialize))
//       .def("deserialize",
//            py::overload_cast<rapidjson::Value &>(&CompositeNode::deserialize),
//            py::arg("json"))
//       .def("deserialize",
//            py::overload_cast<const std::string &>(&CompositeNode::deserialize),
//            py::arg("json_str"))
//       .def("sort_dfs", &CompositeNode::sortDFS);
// }

// }  // namespace dag
// }  // namespace nndeploy