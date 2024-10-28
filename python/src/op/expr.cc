#include "nndeploy/op/expr.h"
#include "nndeploy_api_registry.h"

#include <pybind11/stl.h>

namespace nndeploy {
namespace op {

NNDEPLOY_API_PYBIND11_MODULE("op", m) {
  py::enum_<ExprType>(m, "ExprType")
      .value("kExprTypeValueDesc", kExprTypeValueDesc)
      .value("kExprTypeOpDesc", kExprTypeOpDesc)
      .value("kExprTypeModelDesc", kExprTypeModelDesc)
      .export_values();

  py::class_<op::Expr>(m, "Expr")
      .def(py::init<const std::string &>())
      .def(py::init<const std::string &, base::DataType>())
      .def(py::init<const std::string &, base::DataType, base::IntVector>())
      .def(py::init<std::shared_ptr<ir::ValueDesc>>())
      .def(py::init<std::shared_ptr<ir::OpDesc>>())
      .def(py::init<std::shared_ptr<ir::ModelDesc>>())
      .def("getOutputName", &Expr::getOutputName);

  // 一系列创建函数
  m.def("makeInput", &op::makeInput);
  m.def("makeOutput", &op::makeOutput);
  m.def("makeBlock", &op::makeBlock);
  m.def("makeConv", &op::makeConv);
  m.def("makeRelu", &op::makeRelu);
  m.def("makeSoftMax", &op::makeSoftMax);
  m.def("makeAdd", &op::makeAdd);
}
}  // namespace op
}  // namespace nndeploy