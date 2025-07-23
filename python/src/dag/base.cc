#include "nndeploy/dag/edge.h"

#include "nndeploy_api_registry.h"

namespace nndeploy {

namespace dag {

NNDEPLOY_API_PYBIND11_MODULE("dag", m) {
  py::enum_<NodeType>(m, "NodeType")
      .value("Input", NodeType::kNodeTypeInput)
      .value("Output", NodeType::kNodeTypeOutput)
      .value("Intermediate", NodeType::kNodeTypeIntermediate);

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
#ifdef ENABLE_NNDEPLOY_OPENCV
      .def("set_cvmat_type", &EdgeTypeInfo::setType<cv::Mat>)
#endif
      .def("set_tensor_type", &EdgeTypeInfo::setType<device::Tensor>)
      .def("set_param_type", &EdgeTypeInfo::setType<base::Param>)
      .def("set_type", [](EdgeTypeInfo& self, py::object type_val) {
        if (py::isinstance<py::type>(type_val)) {
          py::type py_type = type_val.cast<py::type>();
          if (py_type.is(py::type::of<device::Buffer>())) {
            self.setType<device::Buffer>();
          } else if (py_type.is(py::type::of<device::Tensor>())) {
            self.setType<device::Tensor>();
          } 
          // else if (py_type.is(py::type::of<base::Param>())) {
          //   self.setType<base::Param>();
          // } 
          else {
            self.type_ = EdgeTypeFlag::kAny;
            // 获取类型的完整名称，包括模块路径
            py::object module = py_type.attr("__module__");
            std::string module_name = module.cast<std::string>();
            std::string type_name = py_type.attr("__name__").cast<std::string>();
            self.type_name_ = module_name + "." + type_name;
            // NNDEPLOY_LOGI("type_name: %s.\n", self.type_name_.c_str());
            self.type_ptr_ = &typeid(py::type);
            self.type_holder_ = std::make_shared<EdgeTypeInfo::TypeHolder<py::type>>();
          }
        } else {
          self.type_ = EdgeTypeFlag::kAny;
          // 获取泛型类型的完整名称，支持泛型内部数据结构解析
          std::string full_type_name;
          
          // 检查是否为泛型类型（如 list[Face]、dict[str, Any] 等）
          if (py::hasattr(type_val, "__origin__") && py::hasattr(type_val, "__args__")) {
            // 处理泛型类型
            py::object origin = type_val.attr("__origin__");
            py::object args = type_val.attr("__args__");
            
            // 获取泛型容器的名称（如 list、dict）
            std::string origin_name = origin.attr("__name__").cast<std::string>();
            full_type_name = origin_name + "[";
            
            // 遍历泛型参数
            py::tuple args_tuple = args.cast<py::tuple>();
            for (size_t i = 0; i < args_tuple.size(); ++i) {
              if (i > 0) full_type_name += ", ";
              
              py::object arg = args_tuple[i];
              if (py::hasattr(arg, "__name__")) {
                // 基本类型或类
                full_type_name += arg.attr("__name__").cast<std::string>();
              } else if (py::hasattr(arg, "_name")) {
                // typing 模块的特殊类型（如 Any）
                full_type_name += arg.attr("_name").cast<std::string>();
              } else {
                // 嵌套泛型或其他复杂类型
                full_type_name += py::str(arg).cast<std::string>();
              }
            }
            full_type_name += "]";
          } else {
            // 普通类型
            py::object type_name = type_val.attr("__name__");
            full_type_name = type_name.cast<std::string>();
          }
          
          self.type_name_ = full_type_name;
          // NNDEPLOY_LOGI("type_name: %s.\n", self.type_name_.c_str());
          self.type_ptr_ = &typeid(py::type);
          self.type_holder_ = std::make_shared<EdgeTypeInfo::TypeHolder<py::type>>();
        }
      }, py::arg("type_val"))
      .def("get_type", &EdgeTypeInfo::getType)
      .def("set_type_name", &EdgeTypeInfo::setTypeName)
      .def("get_type_name", &EdgeTypeInfo::getTypeName)
      .def("get_type_name_with_namespace", &EdgeTypeInfo::getTypeNameWithNamespace)
      .def("get_unique_type_name", &EdgeTypeInfo::getUniqueTypeName)
      .def("get_type_ptr", &EdgeTypeInfo::getTypePtr)
      .def("is_buffer_type", &EdgeTypeInfo::isType<device::Buffer>)
#ifdef ENABLE_NNDEPLOY_OPENCV
      .def("is_cvmat_type", &EdgeTypeInfo::isType<cv::Mat>)
#endif
      .def("is_tensor_type", &EdgeTypeInfo::isType<device::Tensor>)
      .def("is_param_type", &EdgeTypeInfo::isType<base::Param>)
      // .def("create_buffer", &EdgeTypeInfo::createType<device::Buffer>)
      // .def("create_cvmat", &EdgeTypeInfo::createType<cv::Mat>)
      // .def("create_tensor", &EdgeTypeInfo::createType<device::Tensor>)
      // .def("create_param", &EdgeTypeInfo::createType<base::Param>)
      .def("check_buffer_type", &EdgeTypeInfo::checkType<device::Buffer>)
#ifdef ENABLE_NNDEPLOY_OPENCV
      .def("check_cvmat_type", &EdgeTypeInfo::checkType<cv::Mat>)
#endif
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
  // m.def("edge_type_to_string", &edgeTypeToString, py::arg("edge_type"),
  //       "Convert EdgeTypeFlag to string representation");
  // m.def("string_to_edge_type", &stringToEdgeType, py::arg("edge_type_str"),
  //       "Convert string to EdgeTypeFlag");

  py::class_<RunStatus, std::shared_ptr<RunStatus>>(m, "RunStatus")
      .def(py::init<>())
      .def(py::init<const std::string&, bool, size_t, size_t, size_t, float, float>(),
           py::arg("node_name"), py::arg("is_running"), py::arg("graph_run_size"),
           py::arg("run_size"), py::arg("completed_size"), py::arg("cost_time"), py::arg("average_time"))
      .def(py::init<const RunStatus&>())
      .def("get_status", &RunStatus::getStatus)
      .def_readwrite("node_name", &RunStatus::node_name)
      .def_readwrite("is_running", &RunStatus::is_running)
      .def_readwrite("graph_run_size", &RunStatus::graph_run_size)
      .def_readwrite("run_size", &RunStatus::run_size)
      .def_readwrite("completed_size", &RunStatus::completed_size)
      .def_readwrite("cost_time", &RunStatus::cost_time)
      .def_readwrite("average_time", &RunStatus::average_time)
      // 添加to_json方法来得到json对象
      .def("to_json", [](const RunStatus& self) {
        py::dict json_obj;
        json_obj["node_name"] = self.node_name;
        json_obj["is_running"] = self.is_running;
        json_obj["graph_run_size"] = self.graph_run_size;
        json_obj["run_size"] = self.run_size;
        json_obj["completed_size"] = self.completed_size;
        json_obj["cost_time"] = self.cost_time;
        json_obj["average_time"] = self.average_time;
        return json_obj;
      })
      .def("__str__", [](const RunStatus& self) {
        std::string str = "RunStatus(node_name=";
        str += self.node_name;
        str += ", is_running=";
        str += std::to_string(self.is_running);
        str += ", graph_run_size=";
        str += std::to_string(self.graph_run_size);
        str += ", run_size=";
        str += std::to_string(self.run_size);
        str += ", completed_size=";
        str += std::to_string(self.completed_size);
        str += ", cost_time=";
        str += std::to_string(self.cost_time);
        str += ", average_time=";
        str += std::to_string(self.average_time);
        str += ")";
        return str;
      });
}

}  // namespace dag

}  // namespace nndeploy