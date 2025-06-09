#include "nndeploy/preprocess/convert_to.h"

#include "nndeploy_api_registry.h"

namespace py = pybind11;
namespace nndeploy {
namespace preprocess {


NNDEPLOY_API_PYBIND11_MODULE("preprocess", m) {
  py::class_<ConvertToParam, base::Param, std::shared_ptr<ConvertToParam>>(m, "ConvertToParam")
      .def(py::init<>())
      .def_readwrite("dst_data_type_", &ConvertToParam::dst_data_type_);

  py::class_<ConvertTo, dag::Node>(m, "ConvertTo")
      .def(py::init<const std::string &>())
      .def(py::init<const std::string &, std::vector<dag::Edge *>,
                    std::vector<dag::Edge *>>())
      .def("run", &ConvertTo::run);
}

}  // namespace preprocess
}  // namespace nndeploy