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
      .def("set_type", [](EdgeTypeInfo& self, py::object type_val) {
        if (py::isinstance<py::type>(type_val)) {
          py::type py_type = type_val.cast<py::type>();
          if (py_type.is(py::type::of<device::Buffer>())) {
            self.setType<device::Buffer>();
          } else if (py_type.is(py::type::of<device::Tensor>())) {
            self.setType<device::Tensor>();
          } else if (py_type.is(py::type::of<base::Param>())) {
            self.setType<base::Param>();
          } else {
            self.type_ = EdgeTypeFlag::kAny;
            // 获取类型的完整名称，包括模块路径
            py::object module = py_type.attr("__module__");
            std::string module_name = module.cast<std::string>();
            std::string type_name = py_type.attr("__name__").cast<std::string>();
            // std::cout << module_name << " " << type_name << std::endl;
            self.type_name_ = module_name + "." + type_name;
            self.type_ptr_ = &typeid(py::type);
            self.type_holder_ = std::make_shared<EdgeTypeInfo::TypeHolder<py::type>>();
          }
        } else {
          self.type_ = EdgeTypeFlag::kAny;
          // 获取泛型类型的完整名称
          py::object type_name = type_val.attr("__name__");
          // std::cout << "type_name: " << type_name.cast<std::string>() << std::endl;
          self.type_name_ = type_name.cast<std::string>();
          self.type_ptr_ = &typeid(py::type);
          self.type_holder_ = std::make_shared<EdgeTypeInfo::TypeHolder<py::type>>();
        }
      }, py::arg("type_val"))
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