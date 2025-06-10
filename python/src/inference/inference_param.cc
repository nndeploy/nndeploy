#include "nndeploy/inference/inference_param.h"

#include <pybind11/stl.h>

#include "nndeploy_api_registry.h"

namespace py = pybind11;

namespace nndeploy {
namespace inference {

class PyInferenceParam : public InferenceParam {
 public:
  using InferenceParam::InferenceParam;

  std::shared_ptr<base::Param> copy() override {
    PYBIND11_OVERRIDE_NAME(std::shared_ptr<base::Param>, InferenceParam, "copy",
                           copy);
  }

  base::Status copyTo(base::Param *param) override {
    PYBIND11_OVERRIDE_NAME(base::Status, InferenceParam, "copy_to", copyTo,
                           param);
  }

  base::Status set(const std::string &key, base::Any &any) override {
    PYBIND11_OVERRIDE_NAME(base::Status, InferenceParam, "set", set, key, any);
  }

  base::Status get(const std::string &key, base::Any &any) override {
    PYBIND11_OVERRIDE_NAME(base::Status, InferenceParam, "get", get, key, any);
  }

  // base::Status serialize(rapidjson::Value &json,
  //                       rapidjson::Document::AllocatorType &allocator)
  //                       override {
  //   PYBIND11_OVERRIDE_NAME(base::Status, InferenceParam, "serialize",
  //   serialize, json, allocator);
  // }

  // std::string serialize() override {
  //   PYBIND11_OVERRIDE_NAME(base::Status, InferenceParam, "serialize",
  //   serialize, stream);
  // }

  // base::Status saveFile(const std::string &path) override {
  //   PYBIND11_OVERRIDE_NAME(base::Status, InferenceParam, "serialize",
  //   serialize, content, is_file);
  // }

  // base::Status deserialize(rapidjson::Value &json) override {
  //   PYBIND11_OVERRIDE_NAME(base::Status, InferenceParam, "deserialize",
  //   deserialize, json);
  // }

  // base::Status deserialize(const std::string &json_str) override {
  //   PYBIND11_OVERRIDE_NAME(base::Status, InferenceParam, "deserialize",
  //   deserialize, stream);
  // }

  // base::Status loadFile(const std::string &path) override {
  //   PYBIND11_OVERRIDE_NAME(base::Status, InferenceParam, "deserialize",
  //   deserialize, content, is_file);
  // }
};

class PyInferenceParamCreator : public InferenceParamCreator {
 public:
  using InferenceParamCreator::InferenceParamCreator;

  std::shared_ptr<InferenceParam> createInferenceParam(
      base::InferenceType type) override {
    PYBIND11_OVERRIDE_PURE_NAME(std::shared_ptr<InferenceParam>,
                                InferenceParamCreator, "create_inference_param",
                                createInferenceParam, type);
  }
};

NNDEPLOY_API_PYBIND11_MODULE("inference", m) {
  py::class_<InferenceParam, PyInferenceParam, base::Param,
             std::shared_ptr<InferenceParam>>(m, "InferenceParam",
                                              py::dynamic_attr())
      .def(py::init<base::InferenceType>())
      .def_readwrite("inference_type_", &InferenceParam::inference_type_)
      .def_readwrite("model_type_", &InferenceParam::model_type_)
      .def_readwrite("is_path_", &InferenceParam::is_path_)
      .def_property(
          "model_value_",
          [](const InferenceParam &self) {
            return self.model_value_;
          },
          [](InferenceParam &self, const std::vector<std::string> &value) {
            // NNDEPLOY_LOGE("set_model_value: %p", &self);
            // for (auto &v : value) {
            //   NNDEPLOY_LOGE("set_model_value: %s", v.c_str());
            // }
            self.model_value_ = value;
          })
      .def_readwrite("input_num_", &InferenceParam::input_num_)
      .def_readwrite("input_name_", &InferenceParam::input_name_)
      .def_readwrite("input_shape_", &InferenceParam::input_shape_)
      .def_readwrite("output_num_", &InferenceParam::output_num_)
      .def_readwrite("output_name_", &InferenceParam::output_name_)
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
      // export all function
      .def("set_inference_type", &InferenceParam::setInferenceType)
      .def("get_inference_type", &InferenceParam::getInferenceType)
      .def("set_model_type", &InferenceParam::setModelType)
      .def("get_model_type", &InferenceParam::getModelType)
      .def("set_is_path", &InferenceParam::setIsPath)
      .def("get_is_path", &InferenceParam::getIsPath)
      .def("set_model_value",
           py::overload_cast<const std::vector<std::string> &>(
               &InferenceParam::setModelValue))
      .def("set_model_value", py::overload_cast<const std::string &, int>(
                                  &InferenceParam::setModelValue))
      .def("get_model_value", &InferenceParam::getModelValue)
      .def("set_input_num", &InferenceParam::setInputNum)
      .def("get_input_num", &InferenceParam::getInputNum)
      .def("set_input_name",
           py::overload_cast<const std::vector<std::string> &>(
               &InferenceParam::setInputName))
      .def("set_input_name", py::overload_cast<const std::string &, int>(
                                 &InferenceParam::setInputName))
      .def("get_input_name", &InferenceParam::getInputName)
      .def("set_input_shape",
           py::overload_cast<const std::vector<std::vector<int>> &>(
               &InferenceParam::setInputShape))
      .def("set_input_shape", py::overload_cast<const std::vector<int> &, int>(
                                  &InferenceParam::setInputShape))
      .def("get_input_shape", &InferenceParam::getInputShape)
      .def("set_output_num", &InferenceParam::setOutputNum)
      .def("get_output_num", &InferenceParam::getOutputNum)
      .def("set_output_name",
           py::overload_cast<const std::vector<std::string> &>(
               &InferenceParam::setOutputName))
      .def("set_output_name", py::overload_cast<const std::string &, int>(
                                  &InferenceParam::setOutputName))
      .def("get_output_name", &InferenceParam::getOutputName)
      .def("set_encrypt_type", &InferenceParam::setEncryptType)
      .def("get_encrypt_type", &InferenceParam::getEncryptType)
      .def("set_license", &InferenceParam::setLicense)
      .def("get_license", &InferenceParam::getLicense)
      .def("set_device_type", &InferenceParam::setDeviceType)
      .def("get_device_type", &InferenceParam::getDeviceType)
      .def("set_num_thread", &InferenceParam::setNumThread)
      .def("get_num_thread", &InferenceParam::getNumThread)
      .def("set_gpu_tune_kernel", &InferenceParam::setGpuTuneKernel)
      .def("get_gpu_tune_kernel", &InferenceParam::getGpuTuneKernel)
      .def("set_share_memory_mode", &InferenceParam::setShareMemoryMode)
      .def("get_share_memory_mode", &InferenceParam::getShareMemoryMode)
      .def("set_precision_type", &InferenceParam::setPrecisionType)
      .def("get_precision_type", &InferenceParam::getPrecisionType)
      .def("set_power_type", &InferenceParam::setPowerType)
      .def("get_power_type", &InferenceParam::getPowerType)
      .def("set_is_dynamic_shape", &InferenceParam::setIsDynamicShape)
      .def("get_is_dynamic_shape", &InferenceParam::getIsDynamicShape)
      .def("set_min_shape", &InferenceParam::setMinShape)
      .def("get_min_shape", &InferenceParam::getMinShape)
      .def("set_opt_shape", &InferenceParam::setOptShape)
      .def("get_opt_shape", &InferenceParam::getOptShape)
      .def("set_max_shape", &InferenceParam::setMaxShape)
      .def("get_max_shape", &InferenceParam::getMaxShape)
      .def("set_cache_path", &InferenceParam::setCachePath)
      .def("get_cache_path", &InferenceParam::getCachePath)
      .def("set_library_path",
           py::overload_cast<const std::vector<std::string> &>(
               &InferenceParam::setLibraryPath))
      .def("set_library_path", py::overload_cast<const std::string &, int>(
                                   &InferenceParam::setLibraryPath))
      .def("get_library_path", &InferenceParam::getLibraryPath)
      .def("copy", &InferenceParam::copy)
      .def("copy_to", &InferenceParam::copyTo);

  py::class_<InferenceParamCreator, PyInferenceParamCreator,
             std::shared_ptr<InferenceParamCreator>>(m, "InferenceParamCreator")
      .def(py::init<>())
      .def("create_inference_param",
           &InferenceParamCreator::createInferenceParam);

  m.def("register_inference_param_creator",
        [](base::InferenceType type,
           std::shared_ptr<InferenceParamCreator> creator) {
          getGlobalInferenceParamCreatorMap()[type] = creator;
        });

  m.def("create_inference_param", &createInferenceParam, py::arg("type"));
}

}  // namespace inference
}  // namespace nndeploy