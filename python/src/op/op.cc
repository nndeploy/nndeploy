#include "nndeploy_api_registry.h"
#include "op/op_func.h"

namespace nndeploy {
NNDEPLOY_API_PYBIND11_MODULE("op", m) {
  m.def("rms_norm", &rmsNormFunc, py::return_value_policy::take_ownership);
  m.def("batch_norm", &batchNormFunc, py::return_value_policy::take_ownership);
  m.def("relu", &reluFunc, py::return_value_policy::take_ownership);
  m.def("conv", &convFunc, py::return_value_policy::take_ownership);
  m.def("add", &addFunc, py::return_value_policy::take_ownership);
  m.def("flatten", &flattenFunc, py::return_value_policy::take_ownership);
  m.def("gemm", &gemmFunc, py::return_value_policy::take_ownership);
  m.def("global_averagepool", &globalAveragepoolFunc,
        py::return_value_policy::take_ownership);
  m.def("maxpool", &maxPoolFunc, py::return_value_policy::take_ownership);
  m.def("mul", &mulFunc, py::return_value_policy::take_ownership);
}

}  // namespace nndeploy
