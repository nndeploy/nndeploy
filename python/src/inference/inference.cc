
#include "nndeploy/inference/inference.h"

#include <pybind11/stl.h>

#include "nndeploy/inference/inference_param.h"
#include "nndeploy_api_registry.h"

namespace nndeploy {
namespace inference {

class PyInferenceCreator : public InferenceCreator {
 public:
  using InferenceCreator::InferenceCreator;

  Inference* createInference(base::InferenceType type) override {
    PYBIND11_OVERRIDE_PURE(Inference*, InferenceCreator,
                           create_inference_cpp, type);
  }

  std::shared_ptr<Inference> createInferenceSharedPtr(
      base::InferenceType type) override {
    PYBIND11_OVERRIDE_PURE(std::shared_ptr<Inference>,
                           InferenceCreator, create_inference, type);
  }
};

class PyInference : public Inference {
 public:
  using Inference::Inference;

  base::Status init() override {
    PYBIND11_OVERRIDE_PURE(base::Status, Inference, init);
  }

  base::Status deinit() override {
    PYBIND11_OVERRIDE_PURE(base::Status, Inference, deinit);
  }

  base::Status reshape(base::ShapeMap &shape_map) override {
    PYBIND11_OVERRIDE_PURE(base::Status, Inference, reshape, shape_map);
  }

  bool isBatch() override {
    PYBIND11_OVERRIDE(bool, Inference, isBatch);
  }

  bool isShareContext() override {
    PYBIND11_OVERRIDE(bool, Inference, isShareContext);
  }

  bool isShareStream() override {
    PYBIND11_OVERRIDE(bool, Inference, isShareStream);
  }

  bool isInputDynamic() override {
    PYBIND11_OVERRIDE(bool, Inference, isInputDynamic);
  }

  bool isOutputDynamic() override {
    PYBIND11_OVERRIDE(bool, Inference, isOutputDynamic);
  }

  bool canOpInput() override {
    PYBIND11_OVERRIDE(bool, Inference, canOpInput);
  }

  bool canOpOutput() override {
    PYBIND11_OVERRIDE(bool, Inference, canOpOutput);
  }

  base::Status run() override {
    PYBIND11_OVERRIDE_PURE(base::Status, Inference, run);
  }

  device::Tensor *getOutputTensorAfterRun(
      const std::string &name, base::DeviceType device_type, bool is_copy,
      base::DataFormat data_format = base::kDataFormatAuto) override {
    PYBIND11_OVERRIDE_PURE(device::Tensor *, Inference, getOutputTensorAfterRun,
                           name, device_type, is_copy, data_format);
  }
};

NNDEPLOY_API_PYBIND11_MODULE("inference", m) {
  py::class_<Inference, PyInference, std::shared_ptr<Inference>>(m, "Inference")
      .def(py::init<base::InferenceType>())
      .def("get_inference_type", &Inference::getInferenceType)
      .def("set_param_cpp", &Inference::setParam, py::arg("param"))
      .def("set_param", &Inference::setParamSharedPtr, py::arg("param"))
      .def("get_param", &Inference::getParam, py::return_value_policy::reference)
      .def("get_device_type", &Inference::getDeviceType)
      .def("set_stream", &Inference::setStream, py::arg("stream"))
      .def("get_stream", &Inference::getStream, py::return_value_policy::reference)
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
      .def("get_input_tensor_desc", &Inference::getInputTensorDesc, py::arg("name"))
      .def("get_output_tensor_desc", &Inference::getOutputTensorDesc, py::arg("name"))
      .def("get_input_tensor_align_desc", &Inference::getInputTensorAlignDesc, py::arg("name"))
      .def("get_output_tensor_align_desc", &Inference::getOutputTensorAlignDesc, py::arg("name"))
      // 会管理资源吗
      .def("get_all_input_tensor_map", &Inference::getAllInputTensorMap, py::return_value_policy::reference)
      .def("get_all_output_tensor_map", &Inference::getAllOutputTensorMap, py::return_value_policy::reference)
      // 会管理资源吗
      .def("get_all_input_tensor_vector", &Inference::getAllInputTensorVector, py::return_value_policy::reference)
      .def("get_all_output_tensor_vector", &Inference::getAllOutputTensorVector, py::return_value_policy::reference)
      .def("get_input_tensor", &Inference::getInputTensor, py::arg("name"), py::return_value_policy::reference)
      .def("get_output_tensor", &Inference::getOutputTensor, py::arg("name"), py::return_value_policy::reference)
      .def("set_input_tensor", &Inference::setInputTensor, py::arg("name"), py::arg("input_tensor"))
      .def("run", &Inference::run)
      .def("get_output_tensor_after_run", &Inference::getOutputTensorAfterRun, py::arg("name"), py::arg("device_type"), py::arg("is_copy"), py::arg("data_format") = base::kDataFormatAuto);

  py::class_<InferenceCreator, PyInferenceCreator,
             std::shared_ptr<InferenceCreator>>(m, "InferenceCreator")
      .def(py::init<>())
      .def("create_inference_cpp",
           &InferenceCreator::createInference)
      .def("create_inference",
           &InferenceCreator::createInferenceSharedPtr);

  m.def("register_inference_creator",
        [](base::InferenceType type,
           std::shared_ptr<InferenceCreator> creator) {
          getGlobalInferenceCreatorMap()[type] = creator;
        });

  // m.def("create_inference_cpp", &createInference, py::arg("type"));
  m.def("create_inference", &createInferenceSharedPtr,
        py::arg("type"));
}

}  // namespace inference
}  // namespace nndeploy