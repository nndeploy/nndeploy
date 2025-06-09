#include "nndeploy/infer/infer.h"

#include <pybind11/stl.h>

#include "nndeploy_api_registry.h"

namespace nndeploy {
namespace infer {

class PyInfer : public Infer {
 public:
  using Infer::Infer;

  base::Status setInputName(const std::string &name, int index = 0) override {
    PYBIND11_OVERRIDE_NAME(base::Status, Infer, "set_input_name", setInputName,
                           name, index);
  }

  base::Status setOutputName(const std::string &name, int index = 0) override {
    PYBIND11_OVERRIDE_NAME(base::Status, Infer, "set_output_name",
                           setOutputName, name, index);
  }

  base::Status setInputNames(const std::vector<std::string> &names) override {
    PYBIND11_OVERRIDE_NAME(base::Status, Infer, "set_input_names",
                           setInputNames, names);
  }

  base::Status setOutputNames(const std::vector<std::string> &names) override {
    PYBIND11_OVERRIDE_NAME(base::Status, Infer, "set_output_names",
                           setOutputNames, names);
  }

  base::Status setInferenceType(base::InferenceType inference_type) override {
    PYBIND11_OVERRIDE_NAME(base::Status, Infer, "set_inference_type",
                           setInferenceType, inference_type);
  }

  // base::Status setParam(base::Param *param) override {
  //   PYBIND11_OVERRIDE_NAME(base::Status, Infer, "set_param", setParam,
  //   param);
  // }

  // base::Param *getParam() override {
  //   PYBIND11_OVERRIDE_NAME(base::Param *, Infer, "get_param", getParam);
  // }

  base::Status setParamSharedPtr(std::shared_ptr<base::Param> param) override {
    PYBIND11_OVERRIDE_NAME(base::Status, Infer, "set_param", setParamSharedPtr,
                           param);
  }

  std::shared_ptr<base::Param> getParamSharedPtr() override {
    PYBIND11_OVERRIDE_NAME(std::shared_ptr<base::Param>, Infer, "get_param",
                           getParamSharedPtr);
  }

  base::Status init() override {
    PYBIND11_OVERRIDE_NAME(base::Status, Infer, "init", init);
  }

  base::Status deinit() override {
    PYBIND11_OVERRIDE_NAME(base::Status, Infer, "deinit", deinit);
  }

  int64_t getMemorySize() override {
    PYBIND11_OVERRIDE_NAME(int64_t, Infer, "get_memory_size", getMemorySize);
  }

  base::Status setMemory(device::Buffer *buffer) override {
    PYBIND11_OVERRIDE_NAME(base::Status, Infer, "set_memory", setMemory,
                           buffer);
  }

  base::Status run() override {
    PYBIND11_OVERRIDE_NAME(base::Status, Infer, "run", run);
  }

  std::shared_ptr<inference::Inference> getInference() override {
    PYBIND11_OVERRIDE_NAME(std::shared_ptr<inference::Inference>, Infer,
                           "get_inference", getInference);
  }

  // base::Status serialize(
  //     rapidjson::Value &json,
  //     rapidjson::Document::AllocatorType &allocator) override {
  //   PYBIND11_OVERRIDE_NAME(base::Status, Infer, "serialize", serialize, json,
  //                          allocator);
  // }

  // base::Status deserialize(rapidjson::Value &json) override {
  //   PYBIND11_OVERRIDE_NAME(base::Status, Infer, "deserialize", deserialize,
  //                          json);
  // }
};

NNDEPLOY_API_PYBIND11_MODULE("infer", m) {
  py::class_<Infer, dag::Node, PyInfer>(m, "Infer", py::dynamic_attr())
      .def(py::init<const std::string &>())
      .def(py::init<const std::string &, std::vector<dag::Edge *>,
                    std::vector<dag::Edge *>>())
      .def(py::init<const std::string &, base::InferenceType>())
      .def(py::init<const std::string &, std::vector<dag::Edge *>,
                    std::vector<dag::Edge *>, base::InferenceType>())
      .def("set_input_name", &Infer::setInputName, py::arg("name"),
           py::arg("index") = 0)
      .def("set_output_name", &Infer::setOutputName, py::arg("name"),
           py::arg("index") = 0)
      .def("set_input_names", &Infer::setInputNames, py::arg("names"))
      .def("set_output_names", &Infer::setOutputNames, py::arg("names"))
      .def("set_inference_type", &Infer::setInferenceType,
           py::arg("inference_type"))
      .def("set_param", &Infer::setParamSharedPtr, py::arg("param"))
      .def("get_param", &Infer::getParamSharedPtr)
      .def("init", &Infer::init)
      .def("deinit", &Infer::deinit)
      .def("get_memory_size", &Infer::getMemorySize)
      .def("set_memory", &Infer::setMemory, py::arg("buffer"))
      .def("run", &Infer::run)
      .def("get_inference", &Infer::getInference);
      // .def("serialize", &Infer::serialize, py::arg("json"),
      //      py::arg("allocator"))
      // .def("deserialize", &Infer::deserialize, py::arg("json"));

}  // NNDEPLOY_API_PYBIND11_MODULE

}  // namespace infer
}  // namespace nndeploy