#include "nndeploy/op/expr.h"

#include <pybind11/stl.h>

#include "nndeploy_api_registry.h"

namespace nndeploy {
namespace op {

NNDEPLOY_API_PYBIND11_MODULE("op", m) {
  py::enum_<ExprType>(m, "ExprType")
      .value("kExprTypeValueDesc", kExprTypeValueDesc)
      .value("kExprTypeOpDesc", kExprTypeOpDesc)
      .value("kExprTypeModelDesc", kExprTypeModelDesc)
      .export_values();

  py::class_<op::Expr, std::shared_ptr<op::Expr>>(m, "Expr")
      .def(py::init<const std::string &>())
      .def(py::init<const std::string &, base::DataType>())
      .def(py::init<const std::string &, base::DataType, base::IntVector>())
      .def(py::init<std::shared_ptr<ir::ValueDesc>>())
      .def(py::init<std::shared_ptr<ir::OpDesc>>())
      .def(py::init<std::shared_ptr<ir::ModelDesc>>())
      .def("getOutputName", &Expr::getOutputName);

  // 一系列创建函数
  m.def("makeInput", &op::makeInput, py::return_value_policy::reference);
  m.def("makeOutput", &op::makeOutput, py::return_value_policy::reference);
  m.def("makeBlock", &op::makeBlock, py::return_value_policy::reference);
  m.def("makeConv", &op::makeConv, py::return_value_policy::reference);
  m.def("makeRelu", &op::makeRelu, py::return_value_policy::reference);
  m.def("makeSoftMax", &op::makeSoftMax, py::return_value_policy::reference);
  m.def("makeAdd", &op::makeAdd, py::return_value_policy::reference);
  m.def("makeBatchNorm", &op::makeBatchNorm, py::return_value_policy::reference);
}
}  // namespace op
}  // namespace nndeploy