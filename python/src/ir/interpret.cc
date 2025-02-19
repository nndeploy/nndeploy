#include "nndeploy/ir/interpret.h"

#include <pybind11/stl.h>

#include "nndeploy/ir/default_interpret.h"
#include "nndeploy_api_registry.h"

namespace nndeploy {
namespace ir {

class PyInterpret : public Interpret {
 public:
  using Interpret::Interpret;

  base::Status interpret(
      const std::vector<std::string> &model_value,
      const std::vector<ValueDesc> &input = std::vector<ValueDesc>()) {
    PYBIND11_OVERRIDE_PURE(base::Status, Interpret, interpret, model_value,
                           input);
  }
};

class PyInterpretCreator : public InterpretCreator {
 public:
  using InterpretCreator::InterpretCreator;

  Interpret *createInterpret(base::ModelType type) override {
    PYBIND11_OVERRIDE_PURE(Interpret *, InterpretCreator, createInterpret,
                           type);
  }
};

Interpret *convertToInterpretPtr(py::object obj) {
  // 将Python对象转换为Interpret *指针
  Interpret *ptr = obj.cast<Interpret *>();
  return ptr;
}

NNDEPLOY_API_PYBIND11_MODULE("ir", m) {
  py::class_<Interpret, PyInterpret, std::unique_ptr<Interpret>>(m, "Interpret")
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

  py::class_<InterpretCreator, PyInterpretCreator,
             std::shared_ptr<InterpretCreator>>(m, "InterpretCreator")
      .def(py::init<>())
      .def("createInterpret", &InterpretCreator::createInterpret);

  py::class_<DefaultInterpret, Interpret>(m, "DefaultInterpret")
      .def(py::init<>())
      .def("interpret", &DefaultInterpret::interpret);

  m.def("register_interpret_creator",
        [](base::ModelType type, std::shared_ptr<InterpretCreator> creator) {
          getGlobalInterpretCreatorMap()[type] = creator;
          for (auto &iter : getGlobalInterpretCreatorMap()) {
            std::cout << "type: " << iter.first << std::endl;
          }
        });

  m.def("createInterpret", &createInterpret, py::arg("type"),
        py::return_value_policy::reference);

  m.def("convert_to_interpret_ptr", &convertToInterpretPtr);
}

}  // namespace ir
}  // namespace nndeploy