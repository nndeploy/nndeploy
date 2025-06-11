#include "nndeploy/inference/onnxruntime/onnxruntime_inference_param.h"

#include <pybind11/stl.h>

#include "nndeploy_api_registry.h"

namespace py = pybind11;

namespace nndeploy {
namespace inference {

NNDEPLOY_API_PYBIND11_MODULE("inference", m) {
  py::class_<OnnxRuntimeInferenceParam, InferenceParam,
             std::shared_ptr<OnnxRuntimeInferenceParam>>(m, "OnnxRuntimeInferenceParam",
                                              py::dynamic_attr())
      .def(py::init<>())
      .def(py::init<base::InferenceType>())
      .def_readwrite("graph_optimization_level_", &OnnxRuntimeInferenceParam::graph_optimization_level_)
      .def_readwrite("inter_op_num_threads_", &OnnxRuntimeInferenceParam::inter_op_num_threads_)
      .def_readwrite("execution_mode_", &OnnxRuntimeInferenceParam::execution_mode_);
}

}  // namespace inference
}  // namespace nndeploy