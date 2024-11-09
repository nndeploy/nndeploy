#include <pybind11/pybind11.h>

#include "nndeploy_api_registry.h"
namespace py = pybind11;

namespace nndeploy {

PYBIND11_MODULE(_nndeploy_internal, m) {
  nndeploy::NndeployModuleRegistry().ImportAll(m);
}
}  // namespace nndeploy