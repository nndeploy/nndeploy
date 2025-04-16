#include "nndeploy/dag/util.h"

#include "nndeploy_api_registry.h"

namespace nndeploy {
namespace dag {

NNDEPLOY_API_PYBIND11_MODULE("dag", m) {
  py::class_<NodeWrapper>(m, "NodeWrapper", py::dynamic_attr())
      .def(py::init<>())
      .def_readwrite("is_external_", &NodeWrapper::is_external_)
      .def_readwrite("node_", &NodeWrapper::node_,
                     py::return_value_policy::reference)
      .def_readwrite("name_", &NodeWrapper::name_)
      .def_readwrite("predecessors_", &NodeWrapper::predecessors_,
                     py::return_value_policy::reference)
      .def_readwrite("successors_", &NodeWrapper::successors_,
                     py::return_value_policy::reference)
      .def_readwrite("color_", &NodeWrapper::color_);

  py::class_<EdgeWrapper>(m, "EdgeWrapper", py::dynamic_attr())
      .def(py::init<>())
      .def_readwrite("is_external_", &EdgeWrapper::is_external_)
      .def_readwrite("edge_", &EdgeWrapper::edge_,
                     py::return_value_policy::reference)
      .def_readwrite("name_", &EdgeWrapper::name_)
      .def_readwrite("producers_", &EdgeWrapper::producers_,
                     py::return_value_policy::reference)
      .def_readwrite("consumers_", &EdgeWrapper::consumers_,
                     py::return_value_policy::reference);

  m.def("get_edge", &getEdge, py::arg("edge_repository"), py::arg("edge_name"),
        py::return_value_policy::reference);

  m.def("find_edge_wrapper",
        py::overload_cast<std::vector<EdgeWrapper*>&, const std::string&>(
            &findEdgeWrapper),
        py::arg("edge_repository"), py::arg("edge_name"),
        py::return_value_policy::reference);

  m.def("find_edge_wrapper",
        py::overload_cast<std::vector<EdgeWrapper*>&, Edge*>(&findEdgeWrapper),
        py::arg("edge_repository"), py::arg("edge"),
        py::return_value_policy::reference);

  m.def("find_start_edges", &findStartEdges, py::arg("edge_repository"),
        py::return_value_policy::reference);

  m.def("find_end_edges", &findEndEdges, py::arg("edge_repository"),
        py::return_value_policy::reference);

  m.def("get_node", &getNode, py::arg("node_repository"), py::arg("node_name"),
        py::return_value_policy::reference);

  m.def("find_node_wrapper",
        py::overload_cast<std::vector<NodeWrapper*>&, const std::string&>(
            &findNodeWrapper),
        py::arg("node_repository"), py::arg("node_name"),
        py::return_value_policy::reference);

  m.def("find_node_wrapper",
        py::overload_cast<std::vector<NodeWrapper*>&, Node*>(&findNodeWrapper),
        py::arg("node_repository"), py::arg("node"),
        py::return_value_policy::reference);

  m.def("find_start_nodes", &findStartNodes, py::arg("node_repository"),
        py::return_value_policy::reference);

  m.def("find_end_nodes", &findEndNodes, py::arg("node_repository"),
        py::return_value_policy::reference);

  m.def("set_color", &setColor, py::arg("node_repository"), py::arg("color"));

  m.def("dump_dag", &dumpDag, py::arg("edge_repository"),
        py::arg("node_repository"), py::arg("graph_inputs"),
        py::arg("graph_outputs"), py::arg("name"), py::arg("oss"));

  m.def("check_unuse_node", &checkUnuseNode, py::arg("node_repository"),
        py::return_value_policy::reference);

  m.def("check_unuse_edge", &checkUnuseEdge, py::arg("node_repository"),
        py::arg("edge_repository"), py::return_value_policy::reference);

  m.def("topo_sort_bfs", &topoSortBFS, py::arg("node_repository"),
        py::arg("topo_sort_node"));

  m.def("topo_sort_dfs", &topoSortDFS, py::arg("node_repository"),
        py::arg("topo_sort_node"));

  m.def("topo_sort", &topoSort, py::arg("node_repository"),
        py::arg("topo_sort_type"), py::arg("topo_sort_node"));

  m.def("check_edge", &checkEdge, py::arg("src_edges"), py::arg("dst_edges"));
}

}  // namespace dag
}  // namespace nndeploy