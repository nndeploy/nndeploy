#include "nndeploy/ir/interpret.h"

#include <pybind11/stl.h>

#include "nndeploy/ir/default_interpret.h"
#include "nndeploy/ir/onnx/onnx_interpret.h"
#include "nndeploy_api_registry.h"

namespace nndeploy {

namespace ir {

template <class Base = Interpret>
class PyInterpret : public Base {
 public:
  using Base ::Base;

  base::Status interpret(
      const std::vector<std::string> &model_value,
      const std::vector<ValueDesc> &input = std::vector<ValueDesc>()) {
    PYBIND11_OVERRIDE_PURE(base::Status, Base, interpret, model_value, input);
  }
};

template <class Other>
class PyInterpretOther : public PyInterpret<Other> {
 public:
  using PyInterpret<Other>::PyInterpret;

  base::Status interpret(
      const std::vector<std::string> &model_value,
      const std::vector<ValueDesc> &input = std::vector<ValueDesc>()) {
    PYBIND11_OVERRIDE(base::Status, Other, interpret, model_value, input);
  }
};

NNDEPLOY_API_PYBIND11_MODULE("ir", m) {
  m.def("createInterpret", &createInterpret, py::arg("type"),
        py::return_value_policy::reference);

  py::class_<Interpret, PyInterpret<>, std::shared_ptr<Interpret>>(m,
                                                                   "Interpret")
      .def("saveModelToFile", &Interpret::saveModelToFile,
           py::arg("structure_file_path"), py::arg("weight_file_path"))
      .def("dump",
           [](Interpret &self, const std::string &file_path) {
             std::ofstream oss(file_path);
             if (!oss.is_open()) {
               throw std::runtime_error("Failed to open file: " + file_path);
             }

             self.dump(oss);
           })
      .def("getModelDesc", &Interpret::getModelDesc,
           py::return_value_policy::reference);

  py::class_<DefaultInterpret, Interpret, PyInterpretOther<DefaultInterpret>,
             std::shared_ptr<DefaultInterpret>>(m, "DefaultInterpret")
      .def(py::init<>())
      .def("interpret", &DefaultInterpret::interpret, py::arg("model_value"),
           py::arg("input") = std::vector<ValueDesc>());

  py::class_<OnnxInterpret, Interpret, PyInterpretOther<OnnxInterpret>,
             std::shared_ptr<OnnxInterpret>>(m, "OnnxInterpret")
      .def(py::init<>())
      .def("interpret", &OnnxInterpret::interpret, py::arg("model_value"),
           py::arg("input") = std::vector<ValueDesc>());
}

}  // namespace ir

}  // namespace nndeploy