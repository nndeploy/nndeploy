#include "nndeploy/preprocess/cvt_resize_norm_trans.h"

#include "nndeploy_api_registry.h"

namespace py = pybind11;
namespace nndeploy {
namespace preprocess {

NNDEPLOY_API_PYBIND11_MODULE("preprocess", m) {
  py::class_<CvtResizeNormTrans, dag::Node>(m, "CvtResizeNormTrans")
      .def(py::init<const std::string &>())
      .def(py::init<const std::string &, std::vector<dag::Edge *>,
                    std::vector<dag::Edge *>>())
      .def("run", &CvtResizeNormTrans::run);
}

}  // namespace preprocess
}  // namespace nndeploy