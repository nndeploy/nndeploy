#include "nndeploy_api_registry.h"
#include "op/op_func.h"

namespace nndeploy {
NNDEPLOY_API_PYBIND11_MODULE("op", m) { m.def("rms_norm", rmsNormFunc); }

}  // namespace nndeploy
