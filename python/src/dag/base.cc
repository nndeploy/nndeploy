#include "nndeploy/dag/edge.h"

#include "nndeploy_api_registry.h"

namespace nndeploy {

namespace dag {

NNDEPLOY_API_PYBIND11_MODULE("dag", m) {
  py::enum_<EdgeTypeFlag>(m, "EdgeTypeFlag")
      .value("kBuffer", EdgeTypeFlag::kBuffer)
      .value("kCvMat", EdgeTypeFlag::kCvMat)
      .value("kTensor", EdgeTypeFlag::kTensor)
      .value("kParam", EdgeTypeFlag::kParam)
      .value("kAny", EdgeTypeFlag::kAny)
      .value("kNone", EdgeTypeFlag::kNone);

  py::class_<EdgeTypeInfo, std::shared_ptr<EdgeTypeInfo>>(m, "EdgeTypeInfo", py::dynamic_attr())
      .def(py::init<>())
      .def("set_buffer_type", &EdgeTypeInfo::setType<device::Buffer>)
      .def("set_cvmat_type", &EdgeTypeInfo::setType<cv::Mat>)
      .def("set_tensor_type", &EdgeTypeInfo::setType<device::Tensor>)
      .def("set_param_type", &EdgeTypeInfo::setType<base::Param>)
      .def("get_type", &EdgeTypeInfo::getType)
      .def("get_type_ptr", &EdgeTypeInfo::getTypePtr)
      .def("is_buffer_type", &EdgeTypeInfo::isType<device::Buffer>)
      .def("is_cvmat_type", &EdgeTypeInfo::isType<cv::Mat>)
      .def("is_tensor_type", &EdgeTypeInfo::isType<device::Tensor>)
      .def("is_param_type", &EdgeTypeInfo::isType<base::Param>)
      .def_readwrite("type_", &EdgeTypeInfo::type_)
      .def_readwrite("type_name_", &EdgeTypeInfo::type_name_)
      .def_readwrite("type_ptr_", &EdgeTypeInfo::type_ptr_)
      .def_readwrite("type_holder_", &EdgeTypeInfo::type_holder_)
      .def("__str__", [](const EdgeTypeInfo& self) {
        std::string str = "EdgeTypeInfo(type_=";
        str += std::to_string(static_cast<int>(self.type_));
        str += ", type_name_=";
        str += self.type_name_;
        str += ")";
        return str;
      });
}

}  // namespace dag

}  // namespace nndeploy