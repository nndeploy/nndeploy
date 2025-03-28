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
  m.def("makeInput", &makeInput, py::arg("model_desc"), py::arg("name"),
        py::arg("data_type") = nndeploy::base::dataTypeOf<float>(),
        py::arg("shape") = nndeploy::base::IntVector(),
        py::return_value_policy::reference);

  m.def("makeConstant", &makeConstant, py::arg("model_desc"), py::arg("name"),
        py::return_value_policy::reference);

  m.def("makeOutput", &makeOutput, py::arg("model_desc"), py::arg("expr"),
        py::return_value_policy::reference);

  m.def("makeBlock", &makeBlock, py::arg("model_desc"), py::arg("model_block"),
        py::return_value_policy::reference);

  m.def("makeConv", &makeConv, py::arg("model_desc"), py::arg("input"),
        py::arg("param"), py::arg("weight"), py::arg("bias") = "",
        py::arg("op_name") = "", py::arg("output_name") = "",
        py::return_value_policy::reference);

  m.def("makeRelu", &makeRelu, py::arg("model_desc"), py::arg("input"),
        py::arg("op_name") = "", py::arg("output_name") = "",
        py::return_value_policy::reference);

  m.def("makeSigmoid", &makeSigmoid, py::arg("model_desc"), py::arg("input"),
        py::arg("op_name") = "", py::arg("output_name") = "",
        py::return_value_policy::reference);

  m.def("makeSoftmax", &makeSoftmax, py::arg("model_desc"), py::arg("input"),
        py::arg("param"), py::arg("op_name") = "", py::arg("output_name") = "",
        py::return_value_policy::reference);

  m.def("makeBatchNorm", &makeBatchNorm, py::arg("model_desc"),
        py::arg("input"), py::arg("param"), py::arg("scale"), py::arg("bias"),
        py::arg("mean"), py::arg("var"), py::arg("op_name") = "",
        py::arg("output_name") = "", py::return_value_policy::reference);

  m.def("makeAdd", &makeAdd, py::arg("model_desc"), py::arg("input_0"),
        py::arg("input_1"), py::arg("op_name") = "",
        py::arg("output_name") = "", py::return_value_policy::reference);

  m.def("makeGemm", &makeGemm, py::arg("model_desc"), py::arg("input"),
        py::arg("param"), py::arg("weight"), py::arg("bias") = "",
        py::arg("op_name") = "", py::arg("output_name") = "",
        py::return_value_policy::reference);

  m.def("makeFlatten", &makeFlatten, py::arg("model_desc"), py::arg("input"),
        py::arg("param"), py::arg("op_name") = "", py::arg("output_name") = "",
        py::return_value_policy::reference);

  m.def("makeMaxPool", &makeMaxPool, py::arg("model_desc"), py::arg("input"),
        py::arg("param"), py::arg("op_name") = "", py::arg("output_name") = "",
        py::return_value_policy::reference);

  m.def("makeGlobalAveragePool", &makeGlobalAveragePool, py::arg("model_desc"),
        py::arg("input"), py::arg("op_name") = "", py::arg("output_name") = "",
        py::return_value_policy::reference);
}
}  // namespace op
}  // namespace nndeploy