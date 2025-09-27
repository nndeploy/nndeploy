#include "nndeploy/ocr/ocr.h"

#include "nndeploy/ocr/classifier.h"
#include "nndeploy/ocr/detector.h"
#include "nndeploy/ocr/drawbox.h"
#include "nndeploy/ocr/recognizer.h"
#include "nndeploy/ocr/result.h"
#include "nndeploy_api_registry.h"
namespace py = pybind11;

namespace nndeploy {
namespace ocr {

NNDEPLOY_API_PYBIND11_MODULE("ocr", m) {
  py::class_<DetectorPreProcessParam, base::Param,
             std::shared_ptr<DetectorPreProcessParam>>(
      m, "DetectorPreProcessParam")
      .def(py::init<>())
      .def_readwrite("src_pixel_type_",
                     &DetectorPreProcessParam::src_pixel_type_)
      .def_readwrite("dst_pixel_type_",
                     &DetectorPreProcessParam::dst_pixel_type_)
      .def_readwrite("interp_type_", &DetectorPreProcessParam::interp_type_)
      .def_readwrite("data_type_", &DetectorPreProcessParam::data_type_)
      .def_readwrite("data_format_", &DetectorPreProcessParam::data_format_)
      .def_readwrite("h_", &DetectorPreProcessParam::h_)
      .def_readwrite("w_", &DetectorPreProcessParam::w_)
      .def_readwrite("max_side_len_", &DetectorPreProcessParam::max_side_len_)
      .def_readwrite("normalize_", &DetectorPreProcessParam::normalize_)
      .def_readwrite("border_type_", &DetectorPreProcessParam::border_type_)
      .def_readwrite("top_", &DetectorPreProcessParam::top_)
      .def_readwrite("bottom_", &DetectorPreProcessParam::bottom_)
      .def_readwrite("left_", &DetectorPreProcessParam::left_)
      .def_readwrite("right_", &DetectorPreProcessParam::right_)
      .def_property(
          "scale_",
          [](const DetectorPreProcessParam &self) {
            return py::array_t<float>({3}, {sizeof(float)}, self.scale_,
                                      py::cast(self));
          },
          [](DetectorPreProcessParam &self, py::array_t<float> arr) {
            auto buf = arr.request();
            if (buf.ndim != 1 || buf.shape[0] > 4) {
              throw std::runtime_error(
                  "Input array must be 1D with 3 elements");
            }
            float *ptr = static_cast<float *>(buf.ptr);
            for (size_t i = 0; i < buf.shape[0]; i++) {
              self.scale_[i] = ptr[i];
            }
          })
      .def_property(
          "mean_",
          [](const DetectorPreProcessParam &self) {
            return py::array_t<float>({3}, {sizeof(float)}, self.mean_,
                                      py::cast(self));
          },
          [](DetectorPreProcessParam &self, py::array_t<float> arr) {
            auto buf = arr.request();
            if (buf.ndim != 1 || buf.shape[0] > 4) {
              throw std::runtime_error(
                  "Input array must be 1D with 4 elements");
            }
            float *ptr = static_cast<float *>(buf.ptr);
            for (size_t i = 0; i < buf.shape[0]; i++) {
              self.mean_[i] = ptr[i];
            }
          })
      .def_property(
          "std_",
          [](const DetectorPreProcessParam &self) {
            return py::array_t<float>({3}, {sizeof(float)}, self.std_,
                                      py::cast(self));
          },
          [](DetectorPreProcessParam &self, py::array_t<float> arr) {
            auto buf = arr.request();
            if (buf.ndim != 1 || buf.shape[0] > 4) {
              throw std::runtime_error(
                  "Input array must be 1D with 4 elements");
            }
            float *ptr = static_cast<float *>(buf.ptr);
            for (size_t i = 0; i < buf.shape[0]; i++) {
              self.std_[i] = ptr[i];
            }
          })
      .def_property(
          "border_val_",
          [](const DetectorPreProcessParam &self) {
            return py::array_t<double>({3}, {sizeof(double)},
                                       self.border_val_.val_, py::cast(self));
          },
          [](DetectorPreProcessParam &self, py::array_t<double> arr) {
            auto buf = arr.request();
            if (buf.ndim != 1 || buf.shape[0] > 4) {
              throw std::runtime_error(
                  "Input array must be 1D with 4 elements");
            }
            double *ptr = static_cast<double *>(buf.ptr);
            for (size_t i = 0; i < buf.shape[0]; i++) {
              self.border_val_.val_[i] = ptr[i];
            }
          });
  py::class_<DetectorPreProcess, dag::Node>(m, "DetectorPreProcess")
      .def(py::init<const std::string &>())
      .def(py::init<const std::string &, std::vector<dag::Edge *>,
                    std::vector<dag::Edge *>>())
      .def("run", &DetectorPreProcess::run);

  py::class_<DetectorPostParam, base::Param,
             std::shared_ptr<DetectorPostParam>>(m, "DetectorPostParam")
      .def(py::init<>())
      .def_readwrite("det_db_thresh_",
                     &DetectorPostParam::
                         det_db_thresh_)  // 分数阈值，用于决定哪些检测框被保留
      .def_readwrite(
          "det_db_box_thresh_",
          &DetectorPostParam::
              det_db_box_thresh_)  // 非最大抑制(NMS)阈值，用于合并重叠的检测框
      .def_readwrite(
          "det_db_unclip_ratio_",
          &DetectorPostParam::det_db_unclip_ratio_)  // 模型可以识别的类别数量
      .def_readwrite(
          "det_db_score_mode_",
          &DetectorPostParam::det_db_score_mode_)  // 模型输入图像的高度
      .def_readwrite("use_dilation_", &DetectorPostParam::use_dilation_)
      .def_readwrite("version_", &DetectorPostParam::version_);

  // 导出DetectorPostProcess类
  py::class_<DetectorPostProcess, dag::Node>(m, "DetectorPostProcess")
      .def(py::init<const std::string &>())
      .def(py::init<const std::string &, std::vector<dag::Edge *>,
                    std::vector<dag::Edge *>>())
      .def("run", &DetectorPostProcess::run);

  // 导出DetectorGraph类
  py::class_<DetectorGraph, dag::Graph>(m, "DetectorGraph")
      .def(py::init<const std::string &>())
      .def(py::init<const std::string &, std::vector<dag::Edge *>,
                    std::vector<dag::Edge *>>())
      .def("default_param", &DetectorGraph::defaultParam)
      .def("make", &DetectorGraph::make)
      .def("set_inference_type", &DetectorGraph::setInferenceType)
      .def("set_infer_param", &DetectorGraph::setInferParam)
      .def("set_src_pixel_type", &DetectorGraph::setSrcPixelType)
      .def("set_db_thresh", &DetectorGraph::setDbThresh)
      .def("set_db_box_thresh", &DetectorGraph::setDbBoxThresh)
      .def("set_db_unclip_ratio", &DetectorGraph::setDbUnclipRatio)
      .def("set_db_score_mode", &DetectorGraph::setDbScoreMode)
      .def("set_db_use_dilation", &DetectorGraph::setDbUseDilation)
      .def("set_version", &DetectorGraph::setVersion)
      .def("forward", &DetectorGraph::forward,
           py::return_value_policy::reference);

  py::class_<ClassifierPreProcessParam, base::Param,
             std::shared_ptr<ClassifierPreProcessParam>>(
      m, "ClassifierPreProcessParam")
      .def(py::init<>())
      .def_readwrite("src_pixel_type_",
                     &ClassifierPreProcessParam::src_pixel_type_)
      .def_readwrite("dst_pixel_type_",
                     &ClassifierPreProcessParam::dst_pixel_type_)
      .def_readwrite("interp_type_", &ClassifierPreProcessParam::interp_type_)
      .def_readwrite("data_type_", &ClassifierPreProcessParam::data_type_)
      .def_readwrite("data_format_", &ClassifierPreProcessParam::data_format_)
      .def_readwrite("h_", &ClassifierPreProcessParam::h_)
      .def_readwrite("w_", &ClassifierPreProcessParam::w_)
      .def_readwrite("normalize_", &ClassifierPreProcessParam::normalize_)
      .def_readwrite("border_type_", &ClassifierPreProcessParam::border_type_)
      .def_readwrite("top_", &ClassifierPreProcessParam::top_)
      .def_readwrite("bottom_", &ClassifierPreProcessParam::bottom_)
      .def_readwrite("left_", &ClassifierPreProcessParam::left_)
      .def_readwrite("right_", &ClassifierPreProcessParam::right_)
      .def_property(
          "cls_image_shape_",
          [](const ClassifierPreProcessParam &self) {
            return py::array_t<int>(
                {self.cls_image_shape_.size()},  // shape
                {sizeof(int)},                   // strides
                self.cls_image_shape_.data(),    // ✅ 裸指针
                py::cast(&self)  // base: 保证 vector 生命周期
            );
          },
          [](ClassifierPreProcessParam &self, py::array_t<int> arr) {
            auto buf = arr.request();
            if (buf.ndim != 1 || buf.shape[0] != 3) {
              throw std::runtime_error(
                  "Input array must be 1D with exactly 3 elements");
            }
            int *ptr = static_cast<int *>(buf.ptr);
            self.cls_image_shape_.assign(ptr, ptr + buf.shape[0]);
          })
      .def_property(
          "scale_",
          [](const ClassifierPreProcessParam &self) {
            return py::array_t<float>({3}, {sizeof(float)}, self.scale_,
                                      py::cast(self));
          },
          [](ClassifierPreProcessParam &self, py::array_t<float> arr) {
            auto buf = arr.request();
            if (buf.ndim != 1 || buf.shape[0] > 4) {
              throw std::runtime_error(
                  "Input array must be 1D with 3 elements");
            }
            float *ptr = static_cast<float *>(buf.ptr);
            for (size_t i = 0; i < buf.shape[0]; i++) {
              self.scale_[i] = ptr[i];
            }
          })
      .def_property(
          "mean_",
          [](const ClassifierPreProcessParam &self) {
            return py::array_t<float>({3}, {sizeof(float)}, self.mean_,
                                      py::cast(self));
          },
          [](ClassifierPreProcessParam &self, py::array_t<float> arr) {
            auto buf = arr.request();
            if (buf.ndim != 1 || buf.shape[0] > 4) {
              throw std::runtime_error(
                  "Input array must be 1D with 4 elements");
            }
            float *ptr = static_cast<float *>(buf.ptr);
            for (size_t i = 0; i < buf.shape[0]; i++) {
              self.mean_[i] = ptr[i];
            }
          })
      .def_property(
          "std_",
          [](const ClassifierPreProcessParam &self) {
            return py::array_t<float>({3}, {sizeof(float)}, self.std_,
                                      py::cast(self));
          },
          [](ClassifierPreProcessParam &self, py::array_t<float> arr) {
            auto buf = arr.request();
            if (buf.ndim != 1 || buf.shape[0] > 4) {
              throw std::runtime_error(
                  "Input array must be 1D with 4 elements");
            }
            float *ptr = static_cast<float *>(buf.ptr);
            for (size_t i = 0; i < buf.shape[0]; i++) {
              self.std_[i] = ptr[i];
            }
          })
      .def_property(
          "border_val_",
          [](const ClassifierPreProcessParam &self) {
            return py::array_t<double>({3}, {sizeof(double)},
                                       self.border_val_.val_, py::cast(self));
          },
          [](ClassifierPreProcessParam &self, py::array_t<double> arr) {
            auto buf = arr.request();
            if (buf.ndim != 1 || buf.shape[0] > 4) {
              throw std::runtime_error(
                  "Input array must be 1D with 4 elements");
            }
            double *ptr = static_cast<double *>(buf.ptr);
            for (size_t i = 0; i < buf.shape[0]; i++) {
              self.border_val_.val_[i] = ptr[i];
            }
          });
  py::class_<ClassifierPreProcess, dag::Node>(m, "ClassifierPreProcess")
      .def(py::init<const std::string &>())
      .def(py::init<const std::string &, std::vector<dag::Edge *>,
                    std::vector<dag::Edge *>>())
      .def("run", &ClassifierPreProcess::run);

  py::class_<ClassifierPostParam, base::Param,
             std::shared_ptr<ClassifierPostParam>>(m, "ClassifierPostParam")
      .def(py::init<>())
      .def_readwrite("cls_thresh_", &ClassifierPostParam::cls_thresh_)
      .def_readwrite("version_", &ClassifierPostParam::version_);

  // 导出ClassifierPostProcess类
  py::class_<ClassifierPostProcess, dag::Node>(m, "ClassifierPostProcess")
      .def(py::init<const std::string &>())
      .def(py::init<const std::string &, std::vector<dag::Edge *>,
                    std::vector<dag::Edge *>>())
      .def("run", &ClassifierPostProcess::run);

  // 导出ClassifierGraph类
  py::class_<ClassifierGraph, dag::Graph>(m, "ClassifierGraph")
      .def(py::init<const std::string &>())
      .def(py::init<const std::string &, std::vector<dag::Edge *>,
                    std::vector<dag::Edge *>>())
      .def("default_param", &ClassifierGraph::defaultParam)
      .def("make", &ClassifierGraph::make)
      .def("set_inference_type", &ClassifierGraph::setInferenceType)
      .def("set_infer_param", &ClassifierGraph::setInferParam)
      .def("set_src_pixel_type", &ClassifierGraph::setSrcPixelType)
      .def("set_cls_thresh", &ClassifierGraph::setClsThresh)
      .def("set_version", &ClassifierGraph::setVersion)
      .def("forward", &ClassifierGraph::forward,
           py::return_value_policy::reference);

  py::class_<RecognizerPreProcessParam, base::Param,
             std::shared_ptr<RecognizerPreProcessParam>>(
      m, "RecognizerPreProcessParam")
      .def(py::init<>())
      .def_readwrite("src_pixel_type_",
                     &RecognizerPreProcessParam::src_pixel_type_)
      .def_readwrite("dst_pixel_type_",
                     &RecognizerPreProcessParam::dst_pixel_type_)
      .def_readwrite("interp_type_", &RecognizerPreProcessParam::interp_type_)
      .def_readwrite("data_type_", &RecognizerPreProcessParam::data_type_)
      .def_readwrite("data_format_", &RecognizerPreProcessParam::data_format_)
      .def_readwrite("h_", &RecognizerPreProcessParam::h_)
      .def_readwrite("w_", &RecognizerPreProcessParam::w_)
      .def_readwrite("rec_batch_size_",
                     &RecognizerPreProcessParam::rec_batch_size_)
      .def_readwrite("normalize_", &RecognizerPreProcessParam::normalize_)
      .def_readwrite("border_type_", &RecognizerPreProcessParam::border_type_)
      .def_readwrite("top_", &RecognizerPreProcessParam::top_)
      .def_readwrite("bottom_", &RecognizerPreProcessParam::bottom_)
      .def_readwrite("left_", &RecognizerPreProcessParam::left_)
      .def_readwrite("right_", &RecognizerPreProcessParam::right_)
      .def_property(
          "rec_image_shape_",
          [](const RecognizerPreProcessParam &self) {
            return py::array_t<int>({self.rec_image_shape_.size()},
                                    {sizeof(int)}, self.rec_image_shape_.data(),
                                    py::cast(&self));
          },
          [](RecognizerPreProcessParam &self, py::array_t<int> arr) {
            auto buf = arr.request();
            if (buf.ndim != 1 || buf.shape[0] != 3) {
              throw std::runtime_error(
                  "Input array must be 1D with exactly 3 elements");
            }
            int *ptr = static_cast<int *>(buf.ptr);
            self.rec_image_shape_.assign(ptr, ptr + buf.shape[0]);
          })
      .def_property(
          "scale_",
          [](const RecognizerPreProcessParam &self) {
            return py::array_t<float>({3}, {sizeof(float)}, self.scale_,
                                      py::cast(self));
          },
          [](RecognizerPreProcessParam &self, py::array_t<float> arr) {
            auto buf = arr.request();
            if (buf.ndim != 1 || buf.shape[0] > 4) {
              throw std::runtime_error(
                  "Input array must be 1D with 3 elements");
            }
            float *ptr = static_cast<float *>(buf.ptr);
            for (size_t i = 0; i < buf.shape[0]; i++) {
              self.scale_[i] = ptr[i];
            }
          })
      .def_property(
          "mean_",
          [](const RecognizerPreProcessParam &self) {
            return py::array_t<float>({3}, {sizeof(float)}, self.mean_,
                                      py::cast(self));
          },
          [](RecognizerPreProcessParam &self, py::array_t<float> arr) {
            auto buf = arr.request();
            if (buf.ndim != 1 || buf.shape[0] > 4) {
              throw std::runtime_error(
                  "Input array must be 1D with 4 elements");
            }
            float *ptr = static_cast<float *>(buf.ptr);
            for (size_t i = 0; i < buf.shape[0]; i++) {
              self.mean_[i] = ptr[i];
            }
          })
      .def_property(
          "std_",
          [](const RecognizerPreProcessParam &self) {
            return py::array_t<float>({3}, {sizeof(float)}, self.std_,
                                      py::cast(self));
          },
          [](RecognizerPreProcessParam &self, py::array_t<float> arr) {
            auto buf = arr.request();
            if (buf.ndim != 1 || buf.shape[0] > 4) {
              throw std::runtime_error(
                  "Input array must be 1D with 4 elements");
            }
            float *ptr = static_cast<float *>(buf.ptr);
            for (size_t i = 0; i < buf.shape[0]; i++) {
              self.std_[i] = ptr[i];
            }
          })
      .def_property(
          "border_val_",
          [](const RecognizerPreProcessParam &self) {
            return py::array_t<double>({3}, {sizeof(double)},
                                       self.border_val_.val_, py::cast(self));
          },
          [](RecognizerPreProcessParam &self, py::array_t<double> arr) {
            auto buf = arr.request();
            if (buf.ndim != 1 || buf.shape[0] > 4) {
              throw std::runtime_error(
                  "Input array must be 1D with 4 elements");
            }
            double *ptr = static_cast<double *>(buf.ptr);
            for (size_t i = 0; i < buf.shape[0]; i++) {
              self.border_val_.val_[i] = ptr[i];
            }
          });
  py::class_<RecognizerPreProcess, dag::Node>(m, "RecognizerPreProcess")
      .def(py::init<const std::string &>())
      .def(py::init<const std::string &, std::vector<dag::Edge *>,
                    std::vector<dag::Edge *>>())
      .def("run", &RecognizerPreProcess::run);

  py::class_<RecognizerPostParam, base::Param,
             std::shared_ptr<RecognizerPostParam>>(m, "RecognizerPostParam")
      .def(py::init<>())
      .def_readwrite("version_", &RecognizerPostParam::version_)
      .def_readwrite("rec_thresh_", &RecognizerPostParam::rec_thresh_)
      .def_readwrite("character_path_", &RecognizerPostParam::character_path_);

  // 导出RecognizerPostProcess类
  py::class_<RecognizerPostProcess, dag::Node>(m, "RecognizerPostProcess")
      .def(py::init<const std::string &>())
      .def(py::init<const std::string &, std::vector<dag::Edge *>,
                    std::vector<dag::Edge *>>())
      .def("run", &RecognizerPostProcess::run);

  // 导出RecognizerGraph类
  py::class_<RecognizerGraph, dag::Graph>(m, "RecognizerGraph")
      .def(py::init<const std::string &>())
      .def(py::init<const std::string &, std::vector<dag::Edge *>,
                    std::vector<dag::Edge *>>())
      .def("default_param", &RecognizerGraph::defaultParam)
      .def("make", &RecognizerGraph::make)
      .def("set_inference_type", &RecognizerGraph::setInferenceType)
      .def("set_infer_param", &RecognizerGraph::setInferParam)
      .def("set_character_path", &RecognizerGraph::setCharacterPath)
      .def("set_src_pixel_type", &RecognizerGraph::setSrcPixelType)
      .def("set_rec_thresh", &RecognizerGraph::setRecThresh)
      .def("set_version", &RecognizerGraph::setVersion)
      .def("forward", &RecognizerGraph::forward,
           py::return_value_policy::reference);

  py::class_<OCRResult, base::Param, std::shared_ptr<OCRResult>>(m, "OCRResult")
      .def(py::init<>())
      // boxes_
      .def_property(
          "boxes_", [](OCRResult &self) { return py::cast(self.boxes_); },
          [](OCRResult &self, py::list list) {
            self.boxes_.clear();
            for (auto item : list) {
              self.boxes_.push_back(item.cast<std::array<int, 8>>());
            }
          })
      // image_list_
      .def_property(
          "image_list_",
          [](OCRResult &self) {
            py::list py_list;
            for (auto &img : self.image_list_) {
              py_list.append(
                  img);  // 需要先绑定 cv::Mat -> numpy (cv::Mat 转换)
            }
            return py_list;
          },
          [](OCRResult &self, py::list list) {
            self.image_list_.clear();
            for (auto item : list) {
              self.image_list_.push_back(item.cast<cv::Mat>());
            }
          })
      // classifier_result
      .def_property(
          "classifier_result",
          [](OCRResult &self) { return py::cast(self.classifier_result); },
          [](OCRResult &self, py::list list) {
            self.classifier_result.clear();
            for (auto item : list) {
              self.classifier_result.push_back(item.cast<int>());
            }
          })
      // text_
      .def_property(
          "text_", [](OCRResult &self) { return py::cast(self.text_); },
          [](OCRResult &self, py::list list) {
            self.text_.clear();
            for (auto item : list) {
              self.text_.push_back(item.cast<std::string>());
            }
          })
      // rec_scores_
      .def_property(
          "rec_scores_",
          [](OCRResult &self) { return py::cast(self.rec_scores_); },
          [](OCRResult &self, py::list list) {
            self.rec_scores_.clear();
            for (auto item : list) {
              self.rec_scores_.push_back(item.cast<float>());
            }
          })
      // cls_scores_
      .def_property(
          "cls_scores_",
          [](OCRResult &self) { return py::cast(self.cls_scores_); },
          [](OCRResult &self, py::list list) {
            self.cls_scores_.clear();
            for (auto item : list) {
              self.cls_scores_.push_back(item.cast<float>());
            }
          })
      // cls_labels_
      .def_property(
          "cls_labels_",
          [](OCRResult &self) { return py::cast(self.cls_labels_); },
          [](OCRResult &self, py::list list) {
            self.cls_labels_.clear();
            for (auto item : list) {
              self.cls_labels_.push_back(item.cast<int32_t>());
            }
          })
      // table_boxes_
      .def_property(
          "table_boxes_",
          [](OCRResult &self) { return py::cast(self.table_boxes_); },
          [](OCRResult &self, py::list list) {
            self.table_boxes_.clear();
            for (auto item : list) {
              self.table_boxes_.push_back(item.cast<std::array<int, 8>>());
            }
          })
      // table_structure_
      .def_property(
          "table_structure_",
          [](OCRResult &self) { return py::cast(self.table_structure_); },
          [](OCRResult &self, py::list list) {
            self.table_structure_.clear();
            for (auto item : list) {
              self.table_structure_.push_back(item.cast<std::string>());
            }
          })
      // table_html_
      .def_property(
          "table_html_", [](OCRResult &self) { return self.table_html_; },
          [](OCRResult &self, const std::string &val) {
            self.table_html_ = val;
          })
      // detector_resized_w / h
      .def_property(
          "detector_resized_w",
          [](OCRResult &self) { return self.detector_resized_w; },
          [](OCRResult &self, int v) { self.detector_resized_w = v; })
      .def_property(
          "detector_resized_h",
          [](OCRResult &self) { return self.detector_resized_h; },
          [](OCRResult &self, int v) { self.detector_resized_h = v; })
      // clear
      .def("clear", &OCRResult::clear)
      // getText
      .def("getText", &OCRResult::getText);

  py::class_<RotateCropImage, dag::Node>(m, "RotateCropImage")
      .def(py::init<const std::string &>())
      .def(py::init<const std::string &, std::vector<dag::Edge *>,
                    std::vector<dag::Edge *>>())
      .def("run", &RotateCropImage::run);

  py::class_<RotateImage180, dag::Node>(m, "RotateImage180")
      .def(py::init<const std::string &>())
      .def(py::init<const std::string &, std::vector<dag::Edge *>,
                    std::vector<dag::Edge *>>())
      .def("run", &RotateImage180::run);

  py::class_<DrawDetectorBox, dag::Node>(m, "DrawDetectorBox")
      .def(py::init<const std::string &>())
      .def(py::init<const std::string &, std::vector<dag::Edge *>,
                    std::vector<dag::Edge *>>())
      .def("run", &DrawDetectorBox::run);

  py::class_<OcrText, base::Param, std::shared_ptr<OcrText>>(m, "OcrText")
      .def(py::init<>())
      .def_property(
          "texts_", [](OcrText &self) { return py::cast(self.texts_); },
          [](OcrText &self, py::list list) {
            self.texts_.clear();
            for (auto item : list) {
              self.texts_.push_back(item.cast<std::string>());
            }
          });

  py::class_<PrintOcrNodeParam, base::Param,
             std::shared_ptr<PrintOcrNodeParam>>(m, "PrintOcrNodeParam")
      .def(py::init<>())
      .def_readwrite("path_", &PrintOcrNodeParam::path_);

  py::class_<PrintOcrNode, dag::Node>(m, "PrintOcrNode")
      .def(py::init<const std::string &, std::vector<dag::Edge *>,
                    std::vector<dag::Edge *>>())
      .def("set_path", &PrintOcrNode::setPath)
      .def("run", &PrintOcrNode::run);
}

}  // namespace ocr
}  // namespace nndeploy
