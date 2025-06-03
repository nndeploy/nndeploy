#include "nndeploy/dag/edge.h"

#include "nndeploy_api_registry.h"

namespace nndeploy {

namespace dag {

NNDEPLOY_API_PYBIND11_MODULE("dag", m) {
  py::enum_<NodeType>(m, "NodeType")
      .value("kNodeTypeInput", NodeType::kNodeTypeInput)
      .value("kNodeTypeOutput", NodeType::kNodeTypeOutput)
      .value("kNodeTypeIntermediate", NodeType::kNodeTypeIntermediate);

  py::enum_<EdgeTypeFlag>(m, "EdgeTypeFlag")
      .value("kBuffer", EdgeTypeFlag::kBuffer)
      .value("kCvMat", EdgeTypeFlag::kCvMat)
      .value("kTensor", EdgeTypeFlag::kTensor)
      .value("kParam", EdgeTypeFlag::kParam)
      .value("kAny", EdgeTypeFlag::kAny)
      .value("kNone", EdgeTypeFlag::kNone);

  py::class_<EdgeTypeInfo, std::shared_ptr<EdgeTypeInfo>>(m, "EdgeTypeInfo", py::dynamic_attr())
      .def(py::init<>())
      .def(py::init<const EdgeTypeInfo&>())
      .def("set_buffer_type", &EdgeTypeInfo::setType<device::Buffer>)
      .def("set_cvmat_type", &EdgeTypeInfo::setType<cv::Mat>)
      .def("set_tensor_type", &EdgeTypeInfo::setType<device::Tensor>)
      .def("set_param_type", &EdgeTypeInfo::setType<base::Param>)
      .def("get_type", &EdgeTypeInfo::getType)
      .def("set_type_name", &EdgeTypeInfo::setTypeName)
      .def("get_type_name", &EdgeTypeInfo::getTypeName)
      .def("get_unique_type_name", &EdgeTypeInfo::getUniqueTypeName)
      .def("get_type_ptr", &EdgeTypeInfo::getTypePtr)
      .def("is_buffer_type", &EdgeTypeInfo::isType<device::Buffer>)
      .def("is_cvmat_type", &EdgeTypeInfo::isType<cv::Mat>)
      .def("is_tensor_type", &EdgeTypeInfo::isType<device::Tensor>)
      .def("is_param_type", &EdgeTypeInfo::isType<base::Param>)
      // .def("create_buffer", &EdgeTypeInfo::createType<device::Buffer>)
      // .def("create_cvmat", &EdgeTypeInfo::createType<cv::Mat>)
      // .def("create_tensor", &EdgeTypeInfo::createType<device::Tensor>)
      // .def("create_param", &EdgeTypeInfo::createType<base::Param>)
      .def("check_buffer_type", &EdgeTypeInfo::checkType<device::Buffer>)
      .def("check_cvmat_type", &EdgeTypeInfo::checkType<cv::Mat>)
      .def("check_tensor_type", &EdgeTypeInfo::checkType<device::Tensor>)
      .def("check_param_type", &EdgeTypeInfo::checkType<base::Param>)
      .def("set_edge_name", &EdgeTypeInfo::setEdgeName)
      .def("get_edge_name", &EdgeTypeInfo::getEdgeName)
      .def("__eq__", &EdgeTypeInfo::operator==)
      .def("__ne__", &EdgeTypeInfo::operator!=)
      .def_readwrite("type_", &EdgeTypeInfo::type_)
      .def_readwrite("type_name_", &EdgeTypeInfo::type_name_)
      .def_readwrite("type_ptr_", &EdgeTypeInfo::type_ptr_)
      .def_readwrite("type_holder_", &EdgeTypeInfo::type_holder_)
      .def_readwrite("edge_name_", &EdgeTypeInfo::edge_name_)
      .def("__str__", [](const EdgeTypeInfo& self) {
        std::string str = "EdgeTypeInfo(type_=";
        str += std::to_string(static_cast<int>(self.type_));
        str += ", type_name_=";
        str += self.type_name_;
        str += ", edge_name_=";
        str += self.edge_name_;
        str += ")";
        return str;
      });

  m.def("node_type_to_string", &nodeTypeToString, py::arg("node_type"), 
        "Convert NodeType to string representation");
  m.def("string_to_node_type", &stringToNodeType, py::arg("node_type_str"),
        "Convert string to NodeType");
  m.def("edge_type_to_string", &edgeTypeToString, py::arg("edge_type"),
        "Convert EdgeTypeFlag to string representation");
  m.def("string_to_edge_type", &stringToEdgeType, py::arg("edge_type_str"),
        "Convert string to EdgeTypeFlag");
}

}  // namespace dag

}  // namespace nndeploy