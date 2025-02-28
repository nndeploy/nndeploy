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
    PYBIND11_OVERRIDE_PURE_NAME(base::Status, Interpret, "interpret", interpret,
                                model_value, input);
  }
};

class PyInterpretCreator : public InterpretCreator {
 public:
  using InterpretCreator::InterpretCreator;

  Interpret *createInterpret(base::ModelType type,
                             ir::ModelDesc *model_desc = nullptr,
                             bool is_external = false) override {
    PYBIND11_OVERRIDE_PURE_NAME(Interpret *, InterpretCreator,
                                "create_interpret", createInterpret, type,
                                model_desc, is_external);
  }

  std::shared_ptr<Interpret> createInterpretSharedPtr(
      base::ModelType type, ir::ModelDesc *model_desc = nullptr,
      bool is_external = false) override {
    PYBIND11_OVERRIDE_PURE_NAME(std::shared_ptr<Interpret>, InterpretCreator,
                                "create_interpret_shared_ptr",
                                createInterpretSharedPtr, type, model_desc,
                                is_external);
  }
};

NNDEPLOY_API_PYBIND11_MODULE("ir", m) {
  py::class_<Interpret, PyInterpret>(m, "Interpret")
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
      .def("create_interpret", &InterpretCreator::createInterpret,
           py::return_value_policy::take_ownership)
      .def("create_interpret_shared_ptr",
           &InterpretCreator::createInterpretSharedPtr);

  py::class_<DefaultInterpret, Interpret>(m, "DefaultInterpret")
      .def(py::init<>())
      .def("interpret", &DefaultInterpret::interpret);

  m.def("register_interpret_creator",
        [](base::ModelType type, std::shared_ptr<InterpretCreator> creator) {
          getGlobalInterpretCreatorMap()[type] = creator;
          // for (auto &iter : getGlobalInterpretCreatorMap()) {
          //   std::cout << "type: " << iter.first << std::endl;
          // }
        });

  m.def("create_interpret", &createInterpret,
        py::return_value_policy::take_ownership);
  // m.def("create_interpret_shared_ptr", &createInterpretSharedPtr,
  // py::arg("type"),
  //       py::arg("model_desc") = nullptr, py::arg("is_external") = false);
}

}  // namespace ir
}  // namespace nndeploy