
#include "nndeploy/inference/inference.h"

#include <pybind11/stl.h>

#include "nndeploy/inference/inference_param.h"
#include "nndeploy_api_registry.h"

namespace nndeploy {
namespace inference {

class PyInferenceCreator : public InferenceCreator {
 public:
  using InferenceCreator::InferenceCreator;

  std::shared_ptr<Inference> createInference(
      base::InferenceType type) override {
    PYBIND11_OVERRIDE_PURE_NAME(std::shared_ptr<Inference>, InferenceCreator,
                                "create_inference", createInference, type);
  }
};
class PyInference : public Inference {
 public:
  using Inference::Inference;

  base::Status init() override {
    PYBIND11_OVERRIDE_PURE_NAME(base::Status, Inference, "init", init);
  }

  base::Status deinit() override {
    PYBIND11_OVERRIDE_PURE_NAME(base::Status, Inference, "deinit", deinit);
  }

  base::Status reshape(base::ShapeMap &shape_map) override {
    PYBIND11_OVERRIDE_PURE_NAME(base::Status, Inference, "reshape", reshape,
                                shape_map);
  }

  int64_t getMemorySize() override {
    PYBIND11_OVERRIDE_NAME(int64_t, Inference, "get_memory_size",
                           getMemorySize);
  }

  base::Status setMemory(device::Buffer *buffer) override {
    PYBIND11_OVERRIDE_NAME(base::Status, Inference, "set_memory", setMemory,
                           buffer);
  }

  float getGFLOPs() override {
    PYBIND11_OVERRIDE_NAME(float, Inference, "get_gflops", getGFLOPs);
  }

  bool isBatch() override {
    PYBIND11_OVERRIDE_NAME(bool, Inference, "is_batch", isBatch);
  }

  bool isShareContext() override {
    PYBIND11_OVERRIDE_NAME(bool, Inference, "is_share_context", isShareContext);
  }

  bool isShareStream() override {
    PYBIND11_OVERRIDE_NAME(bool, Inference, "is_share_stream", isShareStream);
  }

  bool isInputDynamic() override {
    PYBIND11_OVERRIDE_NAME(bool, Inference, "is_input_dynamic", isInputDynamic);
  }

  bool isOutputDynamic() override {
    PYBIND11_OVERRIDE_NAME(bool, Inference, "is_output_dynamic",
                           isOutputDynamic);
  }

  bool canOpInput() override {
    PYBIND11_OVERRIDE_NAME(bool, Inference, "can_op_input", canOpInput);
  }

  bool canOpOutput() override {
    PYBIND11_OVERRIDE_NAME(bool, Inference, "can_op_output", canOpOutput);
  }

  int getNumOfInputTensor() override {
    PYBIND11_OVERRIDE_NAME(int, Inference, "get_num_of_input_tensor",
                           getNumOfInputTensor);
  }

  int getNumOfOutputTensor() override {
    PYBIND11_OVERRIDE_NAME(int, Inference, "get_num_of_output_tensor",
                           getNumOfOutputTensor);
  }

  std::string getInputName(int i) override {
    PYBIND11_OVERRIDE_NAME(std::string, Inference, "get_input_name",
                           getInputName, i);
  }

  std::string getOutputName(int i) override {
    PYBIND11_OVERRIDE_NAME(std::string, Inference, "get_output_name",
                           getOutputName, i);
  }

  std::vector<std::string> getAllInputTensorName() override {
    PYBIND11_OVERRIDE_NAME(std::vector<std::string>, Inference,
                           "get_all_input_tensor_name", getAllInputTensorName);
  }

  std::vector<std::string> getAllOutputTensorName() override {
    PYBIND11_OVERRIDE_NAME(std::vector<std::string>, Inference,
                           "get_all_output_tensor_name",
                           getAllOutputTensorName);
  }

  base::IntVector getInputShape(const std::string &name) override {
    PYBIND11_OVERRIDE_NAME(base::IntVector, Inference, "get_input_shape",
                           getInputShape, name);
  }

  base::ShapeMap getAllInputShape() override {
    PYBIND11_OVERRIDE_NAME(base::ShapeMap, Inference, "get_all_input_shape",
                           getAllInputShape);
  }

  device::TensorDesc getInputTensorDesc(const std::string &name) override {
    PYBIND11_OVERRIDE_NAME(device::TensorDesc, Inference,
                           "get_input_tensor_desc", getInputTensorDesc, name);
  }

  device::TensorDesc getOutputTensorDesc(const std::string &name) override {
    PYBIND11_OVERRIDE_NAME(device::TensorDesc, Inference,
                           "get_output_tensor_desc", getOutputTensorDesc, name);
  }

  device::TensorDesc getInputTensorAlignDesc(const std::string &name) override {
    PYBIND11_OVERRIDE_NAME(device::TensorDesc, Inference,
                           "get_input_tensor_align_desc",
                           getInputTensorAlignDesc, name);
  }

  device::TensorDesc getOutputTensorAlignDesc(
      const std::string &name) override {
    PYBIND11_OVERRIDE_NAME(device::TensorDesc, Inference,
                           "get_output_tensor_align_desc",
                           getOutputTensorAlignDesc, name);
  }

  std::map<std::string, device::Tensor *> getAllInputTensorMap() override {
    using ReturnType = std::map<std::string, device::Tensor *>;
    PYBIND11_OVERRIDE_NAME(ReturnType, Inference, "get_all_input_tensor_map",
                           getAllInputTensorMap);
  }

  std::map<std::string, device::Tensor *> getAllOutputTensorMap() override {
    using ReturnType = std::map<std::string, device::Tensor *>;
    PYBIND11_OVERRIDE_NAME(ReturnType, Inference, "get_all_output_tensor_map",
                           getAllOutputTensorMap);
  }

  std::vector<device::Tensor *> getAllInputTensorVector() override {
    PYBIND11_OVERRIDE_NAME(std::vector<device::Tensor *>, Inference,
                           "get_all_input_tensor_vector",
                           getAllInputTensorVector);
  }

  std::vector<device::Tensor *> getAllOutputTensorVector() override {
    PYBIND11_OVERRIDE_NAME(std::vector<device::Tensor *>, Inference,
                           "get_all_output_tensor_vector",
                           getAllOutputTensorVector);
  }

  device::Tensor *getInputTensor(const std::string &name) override {
    PYBIND11_OVERRIDE_NAME(device::Tensor *, Inference, "get_input_tensor",
                           getInputTensor, name);
  }

  device::Tensor *getOutputTensor(const std::string &name) override {
    PYBIND11_OVERRIDE_NAME(device::Tensor *, Inference, "get_output_tensor",
                           getOutputTensor, name);
  }

  base::Status setInputTensor(const std::string &name,
                              device::Tensor *input_tensor) override {
    PYBIND11_OVERRIDE_NAME(base::Status, Inference, "set_input_tensor",
                           setInputTensor, name, input_tensor);
  }

  base::Status run() override {
    PYBIND11_OVERRIDE_PURE_NAME(base::Status, Inference, "run", run);
  }

  device::Tensor *getOutputTensorAfterRun(
      const std::string &name, base::DeviceType device_type, bool is_copy,
      base::DataFormat data_format = base::kDataFormatAuto) override {
    PYBIND11_OVERRIDE_PURE_NAME(
        device::Tensor *, Inference, "get_output_tensor_after_run",
        getOutputTensorAfterRun, name, device_type, is_copy, data_format);
  }
};

NNDEPLOY_API_PYBIND11_MODULE("inference", m) {
  py::class_<Inference, PyInference, std::shared_ptr<Inference>>(m, "Inference")
      .def(py::init<base::InferenceType>())
      .def("get_inference_type", &Inference::getInferenceType)
      .def("set_param", &Inference::setParamSharedPtr, py::arg("param"))
      .def("get_param", &Inference::getParamSharedPtr)
      .def("get_device_type", &Inference::getDeviceType)
      .def("set_stream", &Inference::setStream, py::arg("stream"))
      .def("get_stream", &Inference::getStream,
           py::return_value_policy::reference)
      .def("init", &Inference::init)
      .def("deinit", &Inference::deinit)
      .def("get_min_shape", &Inference::getMinShape)
      .def("get_opt_shape", &Inference::getOptShape)
      .def("get_max_shape", &Inference::getMaxShape)
      .def("reshape", &Inference::reshape, py::arg("shape_map"))
      .def("get_memory_size", &Inference::getMemorySize)
      .def("set_memory", &Inference::setMemory, py::arg("buffer"))
      .def("get_gflops", &Inference::getGFLOPs)
      .def("is_batch", &Inference::isBatch)
      .def("is_share_context", &Inference::isShareContext)
      .def("is_share_stream", &Inference::isShareStream)
      .def("is_input_dynamic", &Inference::isInputDynamic)
      .def("is_output_dynamic", &Inference::isOutputDynamic)
      .def("can_op_input", &Inference::canOpInput)
      .def("can_op_output", &Inference::canOpOutput)
      .def("get_num_of_input_tensor", &Inference::getNumOfInputTensor)
      .def("get_num_of_output_tensor", &Inference::getNumOfOutputTensor)
      .def("get_input_name", &Inference::getInputName, py::arg("i") = 0)
      .def("get_output_name", &Inference::getOutputName, py::arg("i") = 0)
      .def("get_all_input_tensor_name", &Inference::getAllInputTensorName)
      .def("get_all_output_tensor_name", &Inference::getAllOutputTensorName)
      .def("get_input_shape", &Inference::getInputShape, py::arg("name"))
      .def("get_all_input_shape", &Inference::getAllInputShape)
      .def("get_input_tensor_desc", &Inference::getInputTensorDesc,
           py::arg("name"))
      .def("get_output_tensor_desc", &Inference::getOutputTensorDesc,
           py::arg("name"))
      .def("get_input_tensor_align_desc", &Inference::getInputTensorAlignDesc,
           py::arg("name"))
      .def("get_output_tensor_align_desc", &Inference::getOutputTensorAlignDesc,
           py::arg("name"))
      // 会管理资源吗
      .def("get_all_input_tensor_map", &Inference::getAllInputTensorMap,
           py::return_value_policy::reference)
      .def("get_all_output_tensor_map", &Inference::getAllOutputTensorMap,
           py::return_value_policy::reference)
      // 会管理资源吗
      .def("get_all_input_tensor_vector", &Inference::getAllInputTensorVector,
           py::return_value_policy::reference)
      .def("get_all_output_tensor_vector", &Inference::getAllOutputTensorVector,
           py::return_value_policy::reference)
      .def("get_input_tensor", &Inference::getInputTensor, py::arg("name"),
           py::return_value_policy::reference)
      .def("get_output_tensor", &Inference::getOutputTensor, py::arg("name"),
           py::return_value_policy::reference)
      .def("set_input_tensor", &Inference::setInputTensor, py::arg("name"),
           py::arg("input_tensor"))
      .def("run", &Inference::run)
      .def("get_output_tensor_after_run", &Inference::getOutputTensorAfterRun,
           py::arg("name"), py::arg("device_type"), py::arg("is_copy"),
           py::arg("data_format") = base::kDataFormatAuto,
           py::return_value_policy::take_ownership);

  py::class_<InferenceCreator, PyInferenceCreator,
             std::shared_ptr<InferenceCreator>>(m, "InferenceCreator")
      .def(py::init<>())
      .def("create_inference", &InferenceCreator::createInference);

  m.def(
      "register_inference_creator",
      [](base::InferenceType type, std::shared_ptr<InferenceCreator> creator) {
        getGlobalInferenceCreatorMap()[type] = creator;
      });

  m.def("create_inference", &createInference, py::arg("type"));
}

}  // namespace inference
}  // namespace nndeploy