#include "nndeploy/preprocess/cvt_resize_crop_norm_trans.h"

#include "nndeploy_api_registry.h"

namespace py = pybind11;
namespace nndeploy {
namespace preprocess {

NNDEPLOY_API_PYBIND11_MODULE("preprocess", m) {
  py::class_<CvtResizeCropNormTrans, dag::Node>(m, "CvtResizeCropNormTrans")
      .def(py::init<const std::string &>())
      .def(py::init<const std::string &, std::vector<dag::Edge *>,
                    std::vector<dag::Edge *>>())
      .def("run", &CvtResizeCropNormTrans::run);
}

}  // namespace preprocess
}  // namespace nndeploy