#include "nndeploy/detect/yolo/yolo.h"
#include "nndeploy_api_registry.h"

namespace py = pybind11;

namespace nndeploy {
namespace detect {

NNDEPLOY_API_PYBIND11_MODULE("detect", m) {
  // 导出YoloPostParam类
  py::class_<YoloPostParam, base::Param, std::shared_ptr<YoloPostParam>>(m, "YoloPostParam")
    .def(py::init<>())
    .def_readwrite("score_threshold_", &YoloPostParam::score_threshold_)
    .def_readwrite("nms_threshold_", &YoloPostParam::nms_threshold_)
    .def_readwrite("num_classes_", &YoloPostParam::num_classes_)
    .def_readwrite("model_h_", &YoloPostParam::model_h_)
    .def_readwrite("model_w_", &YoloPostParam::model_w_)
    .def_readwrite("version_", &YoloPostParam::version_);

  // 导出YoloPostProcess类
  py::class_<YoloPostProcess, dag::Node, std::shared_ptr<YoloPostProcess>>(m, "YoloPostProcess")
    .def(py::init<const std::string&>())
    .def(py::init<const std::string&, std::initializer_list<dag::Edge*>, std::initializer_list<dag::Edge*>>())
    .def(py::init<const std::string&, std::vector<dag::Edge*>, std::vector<dag::Edge*>>())
    .def("run", &YoloPostProcess::run)
    .def("run_v5v6", &YoloPostProcess::runV5V6)
    .def("run_v8", &YoloPostProcess::runV8);

  // 导出YoloGraph类
  py::class_<YoloGraph, dag::Graph, std::shared_ptr<YoloGraph>>(m, "YoloGraph")
    .def(py::init<const std::string&>())
    .def(py::init<const std::string&, std::initializer_list<dag::Edge*>, std::initializer_list<dag::Edge*>>())
    .def(py::init<const std::string&, std::vector<dag::Edge*>, std::vector<dag::Edge*>>())
    .def("make", &YoloGraph::make)
    .def("set_infer_param", &YoloGraph::setInferParam)
    .def("set_src_pixel_type", &YoloGraph::setSrcPixelType)
    .def("set_score_threshold", &YoloGraph::setScoreThreshold)
    .def("set_nms_threshold", &YoloGraph::setNmsThreshold)
    .def("set_num_classes", &YoloGraph::setNumClasses)
    .def("set_model_hw", &YoloGraph::setModelHW)
    .def("set_version", &YoloGraph::setVersion);
}

}  // namespace detect
}  // namespace nndeploy



