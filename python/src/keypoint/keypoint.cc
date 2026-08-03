
#include "nndeploy/keypoint/yolo_pose/yolo_pose.h"
#include "nndeploy_api_registry.h"

namespace py = pybind11;

namespace nndeploy {
namespace keypoint {

NNDEPLOY_API_PYBIND11_MODULE("keypoint", m) {
  py::class_<KeypointKeyPoint>(m, "KeypointKeyPoint")
      .def(py::init<>())
      .def_readwrite("x_", &KeypointKeyPoint::x_)
      .def_readwrite("y_", &KeypointKeyPoint::y_)
      .def_readwrite("confidence_", &KeypointKeyPoint::confidence_);

  py::class_<KpSkeleton>(m, "KpSkeleton")
      .def(py::init<>())
      .def_readwrite("index_", &KpSkeleton::index_)
      .def_readwrite("label_id_", &KpSkeleton::label_id_)
      .def_readwrite("score_", &KpSkeleton::score_)
      .def_readwrite("keypoints_", &KpSkeleton::keypoints_);

  py::class_<KeypointResult, base::Param, std::shared_ptr<KeypointResult>>(
      m, "KeypointResult")
      .def(py::init<>())
      .def_readwrite("skeletons_", &KeypointResult::skeletons_);

  py::class_<KeypointPostParam, base::Param,
             std::shared_ptr<KeypointPostParam>>(m, "KeypointPostParam")
      .def(py::init<>())
      .def_readwrite("score_threshold_", &KeypointPostParam::score_threshold_)
      .def_readwrite("nms_threshold_", &KeypointPostParam::nms_threshold_)
      .def_readwrite("num_classes_", &KeypointPostParam::num_classes_)
      .def_readwrite("num_keypoints_", &KeypointPostParam::num_keypoints_)
      .def_readwrite("model_h_", &KeypointPostParam::model_h_)
      .def_readwrite("model_w_", &KeypointPostParam::model_w_)
      .def_readwrite("version_", &KeypointPostParam::version_);

  py::class_<KeypointPostProcess, dag::Node>(m, "KeypointPostProcess")
      .def(py::init<const std::string&>())
      .def(py::init<const std::string&, std::vector<dag::Edge*>,
                    std::vector<dag::Edge*>>())
      .def("run", &KeypointPostProcess::run);

  py::class_<KeypointGraph, dag::Graph>(m, "KeypointGraph")
      .def(py::init<const std::string&>())
      .def(py::init<const std::string&, std::vector<dag::Edge*>,
                    std::vector<dag::Edge*>>())
      .def("default_param", &KeypointGraph::defaultParam)
      .def("make", &KeypointGraph::make)
      .def("set_inference_type", &KeypointGraph::setInferenceType)
      .def("set_infer_param", &KeypointGraph::setInferParam)
      .def("forward", &KeypointGraph::forward,
           py::return_value_policy::reference);
}

}  // namespace keypoint
}  // namespace nndeploy
