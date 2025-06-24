#include "nndeploy/segment/drawmask.h"

#include "nndeploy_api_registry.h"

namespace py = pybind11;

namespace nndeploy {
namespace segment {

NNDEPLOY_API_PYBIND11_MODULE("segment", m) {
  py::class_<DrawMask, dag::Node>(m, "DrawMask")
      .def(py::init<const std::string &>())
      .def(py::init<const std::string &, std::vector<dag::Edge *>,
                    std::vector<dag::Edge *>>())
      .def("run", &DrawMask::run);
}

}  // namespace segment
}  // namespace nndeploy
