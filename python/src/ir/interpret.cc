#include "nndeploy/ir/interpret.h"

#include <pybind11/stl.h>

#include "nndeploy/ir/default_interpret.h"
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

template <class Base = InterpretCreator>
class PyInterpretCreator : public Base {
 public:
  using Base::Base;

  Interpret *createInterpret(base::ModelType type) override {
    PYBIND11_OVERRIDE_PURE(Interpret *, Base, createInterpret, type);
  }
};

NNDEPLOY_API_PYBIND11_MODULE("ir", m) {
  py::class_<Interpret, PyInterpret<>>(m, "Interpret")
      .def(py::init<>())
      .def("interpret", &Interpret::interpret)
      .def("dump",
           [](Interpret &self, const std::string &file_path) {
             std::ofstream oss(file_path);
             if (!oss.is_open()) {
               throw std::runtime_error("Failed to open file: " + file_path);
             }

             self.dump(oss);
           })
      .def("save_model", &Interpret::saveModel, py::arg("structure_stream"),
           py::arg("st_ptr"))
      .def("save_model_to_file", &Interpret::saveModelToFile,
           py::arg("structure_file_path"), py::arg("weight_file_path"))
      .def("get_model_desc", &Interpret::getModelDesc,
           py::return_value_policy::reference);

  py::class_<InterpretCreator, PyInterpretCreator<>,
             std::shared_ptr<InterpretCreator>>(m, "InterpretCreator")
      .def(py::init<>())
      .def("create_interpret", &InterpretCreator::createInterpret);

  m.def("register_interpret_creator",
        [](base::ModelType type, std::shared_ptr<InterpretCreator> creator) {
          getGlobalInterpretCreatorMap()[type] = creator;
        });

  m.def("create_interpret", &createInterpret, py::arg("type"));
}

}  // namespace ir

}  // namespace nndeploy