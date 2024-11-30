#include "nndeploy/dag/edge.h"

#include "nndeploy_api_registry.h"

namespace nndeploy {

namespace dag {

NNDEPLOY_API_PYBIND11_MODULE("dag", m) {
  py::class_<Edge>(m, "Edge")
      .def(py::init<>())
      .def(py::init<const std::string &>());
}

}  // namespace dag

}  // namespace nndeploy