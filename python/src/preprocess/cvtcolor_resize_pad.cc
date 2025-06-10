#include "nndeploy/preprocess/cvtcolor_resize_pad.h"

#include "nndeploy_api_registry.h"

namespace py = pybind11;
namespace nndeploy {
namespace preprocess {

NNDEPLOY_API_PYBIND11_MODULE("preprocess", m) {
  py::class_<CvtColorResizePad, dag::Node>(m, "CvtColorResizePad")
      .def(py::init<const std::string &>())
      .def(py::init<const std::string &, std::vector<dag::Edge *>,
                    std::vector<dag::Edge *>>())
      .def("run", &CvtColorResizePad::run);
}

}  // namespace preprocess
}  // namespace nndeploy