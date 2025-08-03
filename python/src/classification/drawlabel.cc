#include "nndeploy/classification/drawlabel.h"

#include "nndeploy_api_registry.h"

namespace py = pybind11;

namespace nndeploy {
namespace classification {

NNDEPLOY_API_PYBIND11_MODULE("classification", m) {
  py::class_<DrawLable, dag::Node>(
      m, "DrawLable")
      .def(py::init<const std::string &>())
      .def(py::init<const std::string &, std::vector<dag::Edge *>,
                    std::vector<dag::Edge *>>())
      .def("run", &DrawLable::run);
}

}  // namespace classification
}  // namespace nndeploy
