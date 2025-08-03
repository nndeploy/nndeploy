
#include "nndeploy/segment/result.h"

#include "nndeploy_api_registry.h"

namespace py = pybind11;

namespace nndeploy {
namespace segment {

NNDEPLOY_API_PYBIND11_MODULE("segment", m) {
  py::class_<SegmentResult, base::Param,
             std::shared_ptr<SegmentResult>>(m, "SegmentResult")
      .def(py::init<>())
      .def_readwrite("mask_", &SegmentResult::mask_)
      .def_readwrite("score_", &SegmentResult::score_)
      .def_readwrite("height_", &SegmentResult::height_)
      .def_readwrite("width_", &SegmentResult::width_)
      .def_readwrite("classes_", &SegmentResult::classes_);
}

}  // namespace detect
}  // namespace nndeploy
