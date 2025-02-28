#include "nndeploy/inference/inference_param.h"

#include <pybind11/stl.h>

#include "nndeploy_api_registry.h"

namespace nndeploy {
namespace inference {

class PyInferenceParamCreator : public InferenceParamCreator {
 public:
  using InferenceParamCreator::InferenceParamCreator;

  InferenceParam* createInferenceParam(base::InferenceType type) override {
    PYBIND11_OVERRIDE_PURE_NAME(InferenceParam*, InferenceParamCreator, create_create_inference_param_cpp, type);
  }

  std::shared_ptr<InferenceParam> createInferenceParamSharedPtr(
      base::InferenceType type) override {
    PYBIND11_OVERRIDE_PURE_NAME(std::shared_ptr<InferenceParam>, InferenceParamCreator,
                           create_inference_param, type);
  }
};

NNDEPLOY_API_PYBIND11_MODULE("inference", m) {
  py::class_<InferenceParam, base::Param, std::shared_ptr<InferenceParam>>(
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
      .def("get", &InferenceParam::get);

  py::class_<InferenceParamCreator, PyInferenceParamCreator,
             std::shared_ptr<InferenceParamCreator>>(m, "InferenceParamCreator")
      .def(py::init<>())
      .def("create_inference_param_cpp", &InferenceParamCreator::createInferenceParam)
      .def("create_inference_param",
           &InferenceParamCreator::createInferenceParamSharedPtr);

  m.def("register_inference_param_creator",
        [](base::InferenceType type, std::shared_ptr<InferenceParamCreator> creator) {
          getGlobalInferenceParamCreatorMap()[type] = creator;
        });

  m.def("create_inference_param_cpp", &createInferenceParam, py::arg("type"));
  m.def("create_inference_param", &createInferenceParamSharedPtr,
        py::arg("type"));
}

}  // namespace inference
}  // namespace nndeploy