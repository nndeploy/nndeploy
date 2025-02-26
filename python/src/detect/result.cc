#include "nndeploy/detect/result.h"

#include "nndeploy_api_registry.h"

namespace py = pybind11;

namespace nndeploy {
namespace detect {

NNDEPLOY_API_PYBIND11_MODULE("detect", m) {
  // 首先导入base模块
  // py::module::import("nndeploy._nndeploy_internal.base");

  // 然后再进行DetectBBoxResult和DetectResult的绑定
  py::class_<DetectBBoxResult, base::Param, std::shared_ptr<DetectBBoxResult>>(
      m, "DetectBBoxResult")
      .def(py::init<>())
      .def_readwrite("index_", &DetectBBoxResult::index_)
      .def_readwrite("label_id_", &DetectBBoxResult::label_id_)
      .def_readwrite("score_", &DetectBBoxResult::score_)
      .def_readwrite("bbox_", &DetectBBoxResult::bbox_)
      .def_readwrite("mask_", &DetectBBoxResult::mask_);

  py::class_<DetectResult, base::Param, std::shared_ptr<DetectResult>>(m, "DetectResult")
      .def(py::init<>())
      .def_readwrite("bboxs_", &DetectResult::bboxs_);
}

}  // namespace detect
}  // namespace nndeploy