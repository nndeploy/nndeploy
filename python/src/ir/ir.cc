#include "nndeploy_api_registry.h"

#include "nndeploy/ir/ir.h"

namespace nndeploy {
namespace ir {
NNDEPLOY_API_PYBIND11_MODULE("ir", m) {
  py::class_<ir::ModelDesc>(m, "ModelDesc")
      .def(py::init<>())
      .def_readwrite("name_", &ir::ModelDesc::name_);
}
}  // namespace ir
}  // namespace nndeploy