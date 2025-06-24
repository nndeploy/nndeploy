#include "nndeploy/track/vis_mot.h"

#include "nndeploy_api_registry.h"

namespace py = pybind11;

namespace nndeploy {
namespace track {

NNDEPLOY_API_PYBIND11_MODULE("track", m) {
  py::class_<VisMOT, dag::Node>(m, "VisMOT")
      .def(py::init<const std::string &>())
      .def(py::init<const std::string &, std::vector<dag::Edge *>,
                    std::vector<dag::Edge *>>())
      .def("run", &VisMOT::run);
}

}  // namespace track
}  // namespace nndeploy
