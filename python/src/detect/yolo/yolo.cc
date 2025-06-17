#include "nndeploy/detect/yolo/yolo.h"
#include "nndeploy/detect/yolo/yolox.h"
#include "nndeploy/detect/yolo/yolo_multi_output.h"
#include "nndeploy/detect/yolo/yolo_multi_conv_output.h"

#include "nndeploy_api_registry.h"

namespace py = pybind11;

namespace nndeploy {
namespace detect {

NNDEPLOY_API_PYBIND11_MODULE("detect", m) {
  py::class_<YoloPostParam, base::Param, std::shared_ptr<YoloPostParam>>(
      m, "YoloPostParam")
      .def(py::init<>())
      .def_readwrite("score_threshold_", &YoloPostParam::score_threshold_)
      .def_readwrite("nms_threshold_", &YoloPostParam::nms_threshold_)
      .def_readwrite("num_classes_", &YoloPostParam::num_classes_)
      .def_readwrite("model_h_", &YoloPostParam::model_h_)
      .def_readwrite("model_w_", &YoloPostParam::model_w_)
      .def_readwrite("version_", &YoloPostParam::version_);

  // 导出YoloPostProcess类
  py::class_<YoloPostProcess, dag::Node>(m, "YoloPostProcess")
      .def(py::init<const std::string&>())
      .def(py::init<const std::string&, std::vector<dag::Edge*>,
                    std::vector<dag::Edge*>>())
      .def("run", &YoloPostProcess::run);
  //   .def("run_v5v6", &YoloPostProcess::runV5V6)
  //   .def("run_v8v11", &YoloPostProcess::runV8V11);

  // 导出YoloGraph类
  py::class_<YoloGraph, dag::Graph>(m, "YoloGraph")
      .def(py::init<const std::string&>())
      .def(py::init<const std::string&, std::vector<dag::Edge*>,
                    std::vector<dag::Edge*>>())
      .def("default_param", &YoloGraph::defaultParam)
      .def("make", &YoloGraph::make)
      .def("set_inference_type", &YoloGraph::setInferenceType)
      .def("set_infer_param", &YoloGraph::setInferParam)
      .def("set_src_pixel_type", &YoloGraph::setSrcPixelType)
      .def("set_score_threshold", &YoloGraph::setScoreThreshold)
      .def("set_nms_threshold", &YoloGraph::setNmsThreshold)
      .def("set_num_classes", &YoloGraph::setNumClasses)
      .def("set_model_hw", &YoloGraph::setModelHW)
      .def("set_version", &YoloGraph::setVersion);

  py::class_<YoloXPostParam, base::Param, std::shared_ptr<YoloXPostParam>>(
      m, "YoloXPostParam")
      .def(py::init<>())
      .def_readwrite("score_threshold_", &YoloXPostParam::score_threshold_)  // 分数阈值，用于决定哪些检测框被保留
      .def_readwrite("nms_threshold_", &YoloXPostParam::nms_threshold_)  // 非最大抑制(NMS)阈值，用于合并重叠的检测框 
      .def_readwrite("num_classes_", &YoloXPostParam::num_classes_)  // 模型可以识别的类别数量
      .def_readwrite("model_h_", &YoloXPostParam::model_h_)      // 模型输入图像的高度
      .def_readwrite("model_w_", &YoloXPostParam::model_w_);     // 模型输入图像的宽度

  // 导出YoloPostProcess类
  py::class_<YoloXPostProcess, dag::Node>(m, "YoloXPostProcess")
      .def(py::init<const std::string&>())
      .def(py::init<const std::string&, std::vector<dag::Edge*>,
                    std::vector<dag::Edge*>>())
      .def("run", &YoloXPostProcess::run);

  // 导出YoloGraph类
  py::class_<YoloXGraph, dag::Graph>(m, "YoloXGraph")
      .def(py::init<const std::string&>())
      .def(py::init<const std::string&, std::vector<dag::Edge*>,
                    std::vector<dag::Edge*>>())
      .def("default_param", &YoloXGraph::defaultParam)
      .def("make", &YoloXGraph::make)
      .def("set_inference_type", &YoloXGraph::setInferenceType)
      .def("set_infer_param", &YoloXGraph::setInferParam)
      .def("forward", &YoloXGraph::forward);

  py::class_<YoloMultiOutputPostParam, base::Param, std::shared_ptr<YoloMultiOutputPostParam>>(
      m, "YoloMultiOutputPostParam")
      .def(py::init<>())
      .def_readwrite("score_threshold_", &YoloMultiOutputPostParam::score_threshold_)
      .def_readwrite("nms_threshold_", &YoloMultiOutputPostParam::nms_threshold_)
      .def_readwrite("obj_threshold_", &YoloMultiOutputPostParam::obj_threshold_)
      .def_readwrite("num_classes_", &YoloMultiOutputPostParam::num_classes_)
      .def_readwrite("model_h_", &YoloMultiOutputPostParam::model_h_)
      .def_readwrite("model_w_", &YoloMultiOutputPostParam::model_w_)
      .def_readwrite("version_", &YoloMultiOutputPostParam::version_)
      .def_property("anchors_stride_8",
          [](YoloMultiOutputPostParam& p) { return py::array_t<int>(6, p.anchors_stride_8); },
          [](YoloMultiOutputPostParam& p, py::array_t<int> arr) {
            if(arr.size() != 6) throw std::runtime_error("anchors_stride_8 must have size 6");
            std::memcpy(p.anchors_stride_8, arr.data(), 6 * sizeof(int));
          })
      .def_property("anchors_stride_16",
          [](YoloMultiOutputPostParam& p) { return py::array_t<int>(6, p.anchors_stride_16); },
          [](YoloMultiOutputPostParam& p, py::array_t<int> arr) {
            if(arr.size() != 6) throw std::runtime_error("anchors_stride_16 must have size 6");
            std::memcpy(p.anchors_stride_16, arr.data(), 6 * sizeof(int));
          })
      .def_property("anchors_stride_32",
          [](YoloMultiOutputPostParam& p) { return py::array_t<int>(6, p.anchors_stride_32); },
          [](YoloMultiOutputPostParam& p, py::array_t<int> arr) {
            if(arr.size() != 6) throw std::runtime_error("anchors_stride_32 must have size 6");
            std::memcpy(p.anchors_stride_32, arr.data(), 6 * sizeof(int));
          });

  // 导出YoloMultiOutputPostProcess类
  py::class_<YoloMultiOutputPostProcess, dag::Node>(m, "YoloMultiOutputPostProcess")
      .def(py::init<const std::string&>())
      .def(py::init<const std::string&, std::vector<dag::Edge*>,
                    std::vector<dag::Edge*>>())
      .def("run", &YoloMultiOutputPostProcess::run);

  // 导出YoloMultiOutputGraph类
  py::class_<YoloMultiOutputGraph, dag::Graph>(m, "YoloMultiOutputGraph")
      .def(py::init<const std::string&>())
      .def(py::init<const std::string&, std::vector<dag::Edge*>,
                    std::vector<dag::Edge*>>())
      .def("default_param", &YoloMultiOutputGraph::defaultParam)
      .def("make", &YoloMultiOutputGraph::make)
      .def("set_inference_type", &YoloMultiOutputGraph::setInferenceType)
      .def("set_infer_param", &YoloMultiOutputGraph::setInferParam)
      .def("set_src_pixel_type", &YoloMultiOutputGraph::setSrcPixelType)
      .def("set_score_threshold", &YoloMultiOutputGraph::setScoreThreshold)
      .def("set_nms_threshold", &YoloMultiOutputGraph::setNmsThreshold)
      .def("set_num_classes", &YoloMultiOutputGraph::setNumClasses)
      .def("set_model_hw", &YoloMultiOutputGraph::setModelHW)
      .def("set_version", &YoloMultiOutputGraph::setVersion)
      .def("forward", &YoloMultiOutputGraph::forward);

  py::class_<YoloMultiConvOutputPostParam, base::Param, std::shared_ptr<YoloMultiConvOutputPostParam>>(
      m, "YoloMultiConvOutputPostParam")
      .def(py::init<>())
      .def_readwrite("score_threshold_", &YoloMultiConvOutputPostParam::score_threshold_)
      .def_readwrite("nms_threshold_", &YoloMultiConvOutputPostParam::nms_threshold_)
      .def_readwrite("obj_threshold_", &YoloMultiConvOutputPostParam::obj_threshold_)
      .def_readwrite("num_classes_", &YoloMultiConvOutputPostParam::num_classes_)
      .def_readwrite("model_h_", &YoloMultiConvOutputPostParam::model_h_)
      .def_readwrite("model_w_", &YoloMultiConvOutputPostParam::model_w_)
      .def_readwrite("det_obj_len_", &YoloMultiConvOutputPostParam::det_obj_len_)
      .def_readwrite("det_bbox_len_", &YoloMultiConvOutputPostParam::det_bbox_len_)
      .def_readwrite("det_cls_len_", &YoloMultiConvOutputPostParam::det_cls_len_)
      .def_readwrite("det_len_", &YoloMultiConvOutputPostParam::det_len_)
      .def_readwrite("version_", &YoloMultiConvOutputPostParam::version_)
      .def_property("anchors_",
          [](YoloMultiConvOutputPostParam& p) { 
            std::vector<int> shape = {3, 6};
            return py::array_t<int>(shape, &p.anchors_[0][0]); 
          },
          [](YoloMultiConvOutputPostParam& p, py::array_t<int> arr) {
            if(arr.ndim() != 2 || arr.shape(0) != 3 || arr.shape(1) != 6) 
              throw std::runtime_error("anchors_ must have shape (3,6)");
            std::memcpy(&p.anchors_[0][0], arr.data(), 3 * 6 * sizeof(int));
          })
      .def_property("strides_",
          [](YoloMultiConvOutputPostParam& p) { return py::array_t<int>(3, p.strides_); },
          [](YoloMultiConvOutputPostParam& p, py::array_t<int> arr) {
            if(arr.size() != 3) throw std::runtime_error("strides_ must have size 3");
            std::memcpy(p.strides_, arr.data(), 3 * sizeof(int));
          });

  // 导出YoloMultiConvOutputPostProcess类
  py::class_<YoloMultiConvOutputPostProcess, dag::Node>(m, "YoloMultiConvOutputPostProcess")
      .def(py::init<const std::string&>())
      .def(py::init<const std::string&, std::vector<dag::Edge*>,
                    std::vector<dag::Edge*>>())
      .def("run", &YoloMultiConvOutputPostProcess::run);

  // 导出YoloMultiConvOutputGraph类
  py::class_<YoloMultiConvOutputGraph, dag::Graph>(m, "YoloMultiConvOutputGraph")
      .def(py::init<const std::string&>())
      .def(py::init<const std::string&, std::vector<dag::Edge*>,
                    std::vector<dag::Edge*>>())
      .def("default_param", &YoloMultiConvOutputGraph::defaultParam)
      .def("make", &YoloMultiConvOutputGraph::make)
      .def("set_inference_type", &YoloMultiConvOutputGraph::setInferenceType)
      .def("set_infer_param", &YoloMultiConvOutputGraph::setInferParam)
      .def("set_src_pixel_type", &YoloMultiConvOutputGraph::setSrcPixelType)
      .def("set_score_threshold", &YoloMultiConvOutputGraph::setScoreThreshold)
      .def("set_nms_threshold", &YoloMultiConvOutputGraph::setNmsThreshold)
      .def("set_num_classes", &YoloMultiConvOutputGraph::setNumClasses)
      .def("set_model_hw", &YoloMultiConvOutputGraph::setModelHW)
      .def("set_version", &YoloMultiConvOutputGraph::setVersion)
      .def("forward", &YoloMultiConvOutputGraph::forward);
}

}  // namespace detect
}  // namespace nndeploy
