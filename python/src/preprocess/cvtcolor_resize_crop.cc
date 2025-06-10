#include "nndeploy/preprocess/cvtcolor_resize_crop.h"

#include "nndeploy_api_registry.h"

namespace py = pybind11;
namespace nndeploy {
namespace preprocess {

NNDEPLOY_API_PYBIND11_MODULE("preprocess", m) {
  py::class_<CvtColorResizeCrop, dag::Node>(m, "CvtColorResizeCrop")
      .def(py::init<const std::string &>())
      .def(py::init<const std::string &, std::vector<dag::Edge *>,
                    std::vector<dag::Edge *>>())
      .def("run", &CvtColorResizeCrop::run);
}

}  // namespace preprocess
}  // namespace nndeploy