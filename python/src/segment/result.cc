
#include "nndeploy/segment/result.h"

#include "nndeploy_api_registry.h"

namespace py = pybind11;

namespace nndeploy {
namespace segment {

static device::Tensor *retainTensor(device::Tensor *value) {
  if (!value) return nullptr;
  if (!value->getBuffer() || (!value->isExternalBuffer() &&
      value->getMemoryType() != base::kMemoryTypeExternal)) {
    return new device::Tensor(*value);
  }
  auto result = std::make_unique<device::Tensor>(
      value->getDevice(), value->getDesc(), value->getName());
  if (value->getSize() < result->getSize() ||
      (result->getSize() && (!value->getData() || !result->getData() || !value->copyTo(result.get()))))
    throw std::runtime_error("Could not copy externally backed segment tensor");
  return result.release();
}

NNDEPLOY_API_PYBIND11_MODULE("segment", m) {
  py::class_<SegmentResult, base::Param,
             std::shared_ptr<SegmentResult>>(m, "SegmentResult")
      .def(py::init<>())
      .def_property("mask_",
          [](const SegmentResult &r) { return r.mask_ ? new device::Tensor(*r.mask_) : nullptr; },
          [](SegmentResult &r, device::Tensor *value) {
            auto *replacement = retainTensor(value);
            delete r.mask_;
            r.mask_ = replacement;
          }, py::return_value_policy::take_ownership)
      .def_property("score_",
          [](const SegmentResult &r) { return r.score_ ? new device::Tensor(*r.score_) : nullptr; },
          [](SegmentResult &r, device::Tensor *value) {
            auto *replacement = retainTensor(value);
            delete r.score_;
            r.score_ = replacement;
          }, py::return_value_policy::take_ownership)
      .def_readwrite("height_", &SegmentResult::height_)
      .def_readwrite("width_", &SegmentResult::width_)
      .def_readwrite("classes_", &SegmentResult::classes_);
}

}  // namespace detect
}  // namespace nndeploy
