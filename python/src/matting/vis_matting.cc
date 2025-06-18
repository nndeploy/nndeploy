#include "nndeploy/matting/vis_matting.h"

#include "nndeploy_api_registry.h"

namespace py = pybind11;

namespace nndeploy {
namespace matting {

NNDEPLOY_API_PYBIND11_MODULE("matting", m) {
  py::class_<VisMatting, dag::Node>(m, "VisMatting")
      .def(py::init<const std::string &>())
      .def(py::init<const std::string &, std::vector<dag::Edge *>,
                    std::vector<dag::Edge *>>())
      .def("run", &VisMatting::run);
}

}  // namespace matting
}  // namespace nndeploy
