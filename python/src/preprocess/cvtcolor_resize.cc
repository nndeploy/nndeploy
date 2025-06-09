#include "nndeploy/preprocess/cvtcolor_resize.h"

#include "nndeploy_api_registry.h"

namespace py = pybind11;
namespace nndeploy {
namespace preprocess {

NNDEPLOY_API_PYBIND11_MODULE("preprocess", m) {
  py::class_<CvtColorResize, dag::Node>(m, "CvtColorResize")
      .def(py::init<const std::string &>())
      .def(py::init<const std::string &, std::vector<dag::Edge *>,
                    std::vector<dag::Edge *>>())
      .def("run", &CvtColorResize::run);
}

}  // namespace preprocess
}  // namespace nndeploy