#include "nndeploy/detect/drawbox.h"

#include "nndeploy_api_registry.h"

namespace py = pybind11;

namespace nndeploy {
namespace detect {

NNDEPLOY_API_PYBIND11_MODULE("detect", m) {
  py::class_<DrawBox, dag::Node>(
      m, "DrawBox")
      .def(py::init<const std::string &>())
      .def(py::init<const std::string &, std::vector<dag::Edge *>,
                    std::vector<dag::Edge *>>())
      .def("run", &DrawBox::run);

  py::class_<YoloMultiConvDrawBox, dag::Node>(
      m, "YoloMultiConvDrawBox")
      .def(py::init<const std::string &>())
      .def(py::init<const std::string &, std::vector<dag::Edge *>,
                    std::vector<dag::Edge *>>())
      .def("run", &YoloMultiConvDrawBox::run);
}

}  // namespace detect
}  // namespace nndeploy
