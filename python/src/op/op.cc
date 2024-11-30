#include "nndeploy_api_registry.h"
#include "op/op_func.h"

namespace nndeploy {
NNDEPLOY_API_PYBIND11_MODULE("op", m) {
  m.def("rms_norm", &rmsNormFunc);
  m.def("batch_norm", &batchNormFunc);
  m.def("relu", &reluFunc);
  m.def("conv", &convFunc);
  m.def("add", &addFunc);
  m.def("flatten", &flattenFunc);
  m.def("gemm", &gemmFunc);
  m.def("global_averagepool", &globalAveragepoolFunc);
  m.def("maxpool", &maxPoolFunc);
}

}  // namespace nndeploy
