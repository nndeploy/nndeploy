
#include "nndeploy/matting/result.h"

#include "nndeploy_api_registry.h"

namespace py = pybind11;

namespace nndeploy {
namespace matting {

NNDEPLOY_API_PYBIND11_MODULE("matting", m) {
  py::class_<MattingResult, base::Param, std::shared_ptr<MattingResult>>(
      m, "MattingResult")
      .def(py::init<>())
      .def_readwrite("alpha", &MattingResult::alpha)
      .def_readwrite("foreground", &MattingResult::foreground)
      .def_readwrite("shape", &MattingResult::shape)
      .def_readwrite("contain_foreground", &MattingResult::contain_foreground);
}

}  // namespace matting
}  // namespace nndeploy
