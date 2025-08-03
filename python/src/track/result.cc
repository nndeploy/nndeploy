
#include "nndeploy/track/result.h"

#include "nndeploy_api_registry.h"

namespace py = pybind11;

namespace nndeploy {
namespace track {

NNDEPLOY_API_PYBIND11_MODULE("track", m) {
  py::class_<MOTResult, base::Param, std::shared_ptr<MOTResult>>(m, "MOTResult")
      .def(py::init<>())
      .def_readwrite("boxes", &MOTResult::boxes)
      .def_readwrite("ids", &MOTResult::ids)
      .def_readwrite("scores", &MOTResult::scores)
      .def_readwrite("class_ids", &MOTResult::class_ids);
}

}  // namespace track
}  // namespace nndeploy
