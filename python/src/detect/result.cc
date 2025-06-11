#include "nndeploy/detect/result.h"

#include "nndeploy_api_registry.h"

namespace py = pybind11;

namespace nndeploy {
namespace detect {

NNDEPLOY_API_PYBIND11_MODULE("detect", m) {
  py::class_<DetectBBoxResult, base::Param, std::shared_ptr<DetectBBoxResult>>(
      m, "DetectBBoxResult")
      .def(py::init<>())
      .def_readwrite("index_", &DetectBBoxResult::index_)
      .def_readwrite("label_id_", &DetectBBoxResult::label_id_)
      .def_readwrite("score_", &DetectBBoxResult::score_)
      .def_property(
          "bbox_",
          [](const DetectBBoxResult& self) {
            return py::array_t<float>({4},                // shape
                                      {sizeof(float)},    // strides
                                      self.bbox_.data(),  // data pointer
                                      py::cast(self)      // parent object
            );
          },
          [](DetectBBoxResult& self, py::array_t<float> arr) {
            auto buf = arr.request();
            if (buf.ndim != 1 || buf.shape[0] > 4) {
              throw std::runtime_error(
                  "Input array must be 1D with 4 elements");
            }
            float* ptr = static_cast<float*>(buf.ptr);
            for (size_t i = 0; i < buf.shape[0]; i++) {
              self.bbox_[i] = ptr[i];
            }
          })
      .def_readwrite("mask_", &DetectBBoxResult::mask_);

  py::class_<DetectResult, base::Param, std::shared_ptr<DetectResult>>(
      m, "DetectResult")
      .def(py::init<>())
      .def_property(
          "bboxs_", [](DetectResult& self) { return py::cast(self.bboxs_); },
          [](DetectResult& self, py::list list) {
            self.bboxs_.clear();
            for (auto item : list) {
              self.bboxs_.push_back(item.cast<DetectBBoxResult>());
            }
          });
}

}  // namespace detect
}  // namespace nndeploy