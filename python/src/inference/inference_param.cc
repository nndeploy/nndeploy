#include "nndeploy/inference/inference_param.h"

#include <pybind11/stl.h>

#include "nndeploy_api_registry.h"

namespace nndeploy {
namespace inference {

class PyInferenceParam : public InferenceParam {
 public:
  using InferenceParam::InferenceParam;

  std::shared_ptr<base::Param> copy() override {
    PYBIND11_OVERRIDE_NAME(std::shared_ptr<base::Param>, InferenceParam, "copy", copy);
  }

  base::Status copyTo(base::Param *param) override {
    PYBIND11_OVERRIDE_NAME(base::Status, InferenceParam, "copy_to", copyTo, param);
  }

  base::Status set(const std::string &key, base::Any &any) override {
    PYBIND11_OVERRIDE_NAME(base::Status, InferenceParam, "set", set, key, any);
  }

  base::Status get(const std::string &key, base::Any &any) override {
    PYBIND11_OVERRIDE_NAME(base::Status, InferenceParam, "get", get, key, any);
  }

  // base::Status serialize(rapidjson::Value &json,
  //                       rapidjson::Document::AllocatorType &allocator) override {
  //   PYBIND11_OVERRIDE_NAME(base::Status, InferenceParam, "serialize", serialize, json, allocator);
  // }

  // base::Status serialize(std::ostream &stream) override {
  //   PYBIND11_OVERRIDE_NAME(base::Status, InferenceParam, "serialize", serialize, stream);
  // }

  // base::Status serialize(std::string &content, bool is_file) override {
  //   PYBIND11_OVERRIDE_NAME(base::Status, InferenceParam, "serialize", serialize, content, is_file);
  // }

  // base::Status deserialize(rapidjson::Value &json) override {
  //   PYBIND11_OVERRIDE_NAME(base::Status, InferenceParam, "deserialize", deserialize, json);
  // }

  // base::Status deserialize(std::istream &stream) override {
  //   PYBIND11_OVERRIDE_NAME(base::Status, InferenceParam, "deserialize", deserialize, stream);
  // }

  // base::Status deserialize(const std::string &content, bool is_file) override {
  //   PYBIND11_OVERRIDE_NAME(base::Status, InferenceParam, "deserialize", deserialize, content, is_file);
  // }
};

class PyInferenceParamCreator : public InferenceParamCreator {
 public:
  using InferenceParamCreator::InferenceParamCreator;

  std::shared_ptr<InferenceParam> createInferenceParam(
      base::InferenceType type) override {
    PYBIND11_OVERRIDE_PURE_NAME(std::shared_ptr<InferenceParam>, InferenceParamCreator,
                           "create_inference_param", createInferenceParam, type);
  }
};

NNDEPLOY_API_PYBIND11_MODULE("inference", m) {
  py::class_<InferenceParam, PyInferenceParam, base::Param, std::shared_ptr<InferenceParam>>(
      m, "InferenceParam", py::dynamic_attr())
      .def(py::init<base::InferenceType>())
      .def_readwrite("inference_type_", &InferenceParam::inference_type_)
      .def_readwrite("model_type_", &InferenceParam::model_type_)
      .def_readwrite("is_path_", &InferenceParam::is_path_)
      .def_readwrite("model_value_", &InferenceParam::model_value_)
      .def_readwrite("encrypt_type_", &InferenceParam::encrypt_type_)
      .def_readwrite("license_", &InferenceParam::license_)
      .def_readwrite("device_type_", &InferenceParam::device_type_)
      .def_readwrite("num_thread_", &InferenceParam::num_thread_)
      .def_readwrite("gpu_tune_kernel_", &InferenceParam::gpu_tune_kernel_)
      .def_readwrite("share_memory_mode_", &InferenceParam::share_memory_mode_)
      .def_readwrite("precision_type_", &InferenceParam::precision_type_)
      .def_readwrite("power_type_", &InferenceParam::power_type_)
      .def_readwrite("is_dynamic_shape_", &InferenceParam::is_dynamic_shape_)
      .def_readwrite("min_shape_", &InferenceParam::min_shape_)
      .def_readwrite("opt_shape_", &InferenceParam::opt_shape_)
      .def_readwrite("max_shape_", &InferenceParam::max_shape_)
      .def_readwrite("cache_path_", &InferenceParam::cache_path_)
      .def_readwrite("library_path_", &InferenceParam::library_path_)
      .def("set", &InferenceParam::set)
      .def("get", &InferenceParam::get)
      .def("copy", &InferenceParam::copy)
      .def("copy_to", &InferenceParam::copyTo);

  py::class_<InferenceParamCreator, PyInferenceParamCreator,
             std::shared_ptr<InferenceParamCreator>>(m, "InferenceParamCreator")
      .def(py::init<>())
      .def("create_inference_param",
           &InferenceParamCreator::createInferenceParam);

  m.def("register_inference_param_creator",
        [](base::InferenceType type, std::shared_ptr<InferenceParamCreator> creator) {
          getGlobalInferenceParamCreatorMap()[type] = creator;
        });

  m.def("create_inference_param", &createInferenceParam, py::arg("type"));
}

}  // namespace inference
}  // namespace nndeploy