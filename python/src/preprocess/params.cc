
#ifdef _WIN32
#include <BaseTsd.h>
typedef SSIZE_T ssize_t;
#endif

#include "nndeploy/preprocess/params.h"

#include "nndeploy_api_registry.h"

namespace py = pybind11;
namespace nndeploy {
namespace preprocess {


NNDEPLOY_API_PYBIND11_MODULE("preprocess", m) {
  py::class_<CvtcolorParam, base::Param, std::shared_ptr<CvtcolorParam>>(m, "CvtcolorParam")
      .def(py::init<>())
      .def_readwrite("src_pixel_type_", &CvtcolorParam::src_pixel_type_)
      .def_readwrite("dst_pixel_type_", &CvtcolorParam::dst_pixel_type_);

  py::class_<CropParam, base::Param, std::shared_ptr<CropParam>>(m, "CropParam")
      .def(py::init<>())
      .def_readwrite("top_left_x_", &CropParam::top_left_x_)
      .def_readwrite("top_left_y_", &CropParam::top_left_y_)
      .def_readwrite("width_", &CropParam::width_)
      .def_readwrite("height_", &CropParam::height_);

  py::class_<NormlizeParam, base::Param, std::shared_ptr<NormlizeParam>>(m, "NormlizeParam")
      .def(py::init<>())
      .def_property(
          "scale_",
          [](const NormlizeParam& self) {
            return py::array_t<float>({4},              // shape
                                      {sizeof(float)},  // strides
                                      self.scale_,      // data pointer
                                      py::cast(self)    // parent object
            );
          },
          [](NormlizeParam& self, py::array_t<float> arr) {
            auto buf = arr.request();
            if (buf.ndim != 1 || buf.shape[0] > 4) {
              throw std::runtime_error(
                  "Input array must be 1D with 4 elements");
            }
            float* ptr = static_cast<float*>(buf.ptr);
            for (size_t i = 0; i < buf.shape[0]; i++) {
              self.scale_[i] = ptr[i];
            }
          })
      .def_property(
          "mean_",
          [](const NormlizeParam& self) {
            return py::array_t<float>({4},              // shape
                                      {sizeof(float)},  // strides
                                      self.mean_,       // data pointer
                                      py::cast(self)    // parent object
            );
          },
          [](NormlizeParam& self, py::array_t<float> arr) {
            auto buf = arr.request();
            if (buf.ndim != 1 || buf.shape[0] > 4) {
              throw std::runtime_error(
                  "Input array must be 1D with 4 elements");
            }
            float* ptr = static_cast<float*>(buf.ptr);
            for (size_t i = 0; i < buf.shape[0]; i++) {
              self.mean_[i] = ptr[i];
            }
          })
      .def_property(
          "std_",
          [](const NormlizeParam& self) {
            return py::array_t<float>({4},              // shape
                                      {sizeof(float)},  // strides
                                      self.std_,        // data pointer
                                      py::cast(self)    // parent object
            );
          },
          [](NormlizeParam& self, py::array_t<float> arr) {
            auto buf = arr.request();
            if (buf.ndim != 1 || buf.shape[0] > 4) {
              throw std::runtime_error(
                  "Input array must be 1D with 4 elements");
            }
            float* ptr = static_cast<float*>(buf.ptr);
            for (size_t i = 0; i < buf.shape[0]; i++) {
              self.std_[i] = ptr[i];
            }
          });

  py::class_<TransposeParam, base::Param, std::shared_ptr<TransposeParam>>(m,
                                                              "TransposeParam")
      .def(py::init<>())
      .def_readwrite("src_data_format_", &TransposeParam::src_data_format_)
      .def_readwrite("dst_data_format_", &TransposeParam::dst_data_format_);

  py::class_<DynamicShapeParam, base::Param, std::shared_ptr<DynamicShapeParam>>(
      m, "DynamicShapeParam")
      .def(py::init<>())
      .def_readwrite("is_power_of_n_", &DynamicShapeParam::is_power_of_n_)
      .def_readwrite("n_", &DynamicShapeParam::n_)
      .def_readwrite("w_align_", &DynamicShapeParam::w_align_)
      .def_readwrite("h_align_", &DynamicShapeParam::h_align_)
      .def_property(
          "min_shape_",
          [](const DynamicShapeParam& self) {
            return py::array_t<int>({self.min_shape_.size()}, {sizeof(int)},
                                    self.min_shape_.data(), py::cast(self));
          },
          [](DynamicShapeParam& self, py::array_t<int> arr) {
            auto buf = arr.request();
            int* ptr = static_cast<int*>(buf.ptr);
            self.min_shape_.assign(ptr, ptr + buf.shape[0]);
          })
      .def_property(
          "opt_shape_",
          [](const DynamicShapeParam& self) {
            return py::array_t<int>({self.opt_shape_.size()}, {sizeof(int)},
                                    self.opt_shape_.data(), py::cast(self));
          },
          [](DynamicShapeParam& self, py::array_t<int> arr) {
            auto buf = arr.request();
            int* ptr = static_cast<int*>(buf.ptr);
            self.opt_shape_.assign(ptr, ptr + buf.shape[0]);
          })
      .def_property(
          "max_shape_",
          [](const DynamicShapeParam& self) {
            return py::array_t<int>({self.max_shape_.size()}, {sizeof(int)},
                                    self.max_shape_.data(), py::cast(self));
          },
          [](DynamicShapeParam& self, py::array_t<int> arr) {
            auto buf = arr.request();
            int* ptr = static_cast<int*>(buf.ptr);
            self.max_shape_.assign(ptr, ptr + buf.shape[0]);
          });

  py::class_<ResizeParam, base::Param, std::shared_ptr<ResizeParam>>(m, "ResizeParam")
      .def(py::init<>())
      .def_readwrite("interp_type_", &ResizeParam::interp_type_)
      .def_readwrite("scale_w_", &ResizeParam::scale_w_)
      .def_readwrite("scale_h_", &ResizeParam::scale_h_)
      .def_readwrite("dst_h_", &ResizeParam::dst_h_)
      .def_readwrite("dst_w_", &ResizeParam::dst_w_);

  py::class_<PaddingParam, base::Param, std::shared_ptr<PaddingParam>>(m, "PaddingParam")
      .def(py::init<>())
      .def_readwrite("border_type_", &PaddingParam::border_type_)
      .def_readwrite("top_", &PaddingParam::top_)
      .def_readwrite("bottom_", &PaddingParam::bottom_)
      .def_readwrite("left_", &PaddingParam::left_)
      .def_readwrite("right_", &PaddingParam::right_)
      .def_property(
          "border_val_",
          [](const PaddingParam& self) {
            return py::array_t<double>({4},                    // shape
                                       {sizeof(double)},       // strides
                                       self.border_val_.val_,  // data pointer
                                       py::cast(self)          // parent object
            );
          },
          [](PaddingParam& self, py::array_t<double> arr) {
            auto buf = arr.request();
            if (buf.ndim != 1 || buf.shape[0] > 4) {
              throw std::runtime_error(
                  "Input array must be 1D with 4 elements");
            }
            double* ptr = static_cast<double*>(buf.ptr);
            for (size_t i = 0; i < buf.shape[0]; i++) {
              self.border_val_.val_[i] = ptr[i];
            }
            for (size_t i = buf.shape[0]; i < 4; i++) {
              self.border_val_.val_[i] = 0.0;
            }
          });

  py::class_<WarpAffineCvtNormTransParam, base::Param, std::shared_ptr<WarpAffineCvtNormTransParam>>(
      m, "WarpAffineCvtNormTransParam")
      .def(py::init<>())
      .def_property(
          "transform_",
          [](const WarpAffineCvtNormTransParam& self) {
            return py::array_t<float>(
                std::vector<ssize_t>{2, 3},  // shape
                std::vector<ssize_t>{sizeof(float) * 3,
                                     sizeof(float)},  // strides
                &self.transform_[0][0],               // data pointer
                py::cast(self)                        // parent object
            );
          },
          [](WarpAffineCvtNormTransParam& self, py::array_t<float> arr) {
            auto buf = arr.request();
            if (buf.ndim != 2 || buf.shape[0] != 2 || buf.shape[1] != 3) {
              throw std::runtime_error(
                  "Input array must be 2D with shape (2, 3)");
            }
            float* ptr = static_cast<float*>(buf.ptr);
            for (size_t i = 0; i < 2; i++) {
              for (size_t j = 0; j < 3; j++) {
                self.transform_[i][j] = ptr[i * 3 + j];
              }
            }
          })
      .def_readwrite("dst_w_", &WarpAffineCvtNormTransParam::dst_w_)
      .def_readwrite("dst_h_", &WarpAffineCvtNormTransParam::dst_h_)
      .def_readwrite("src_pixel_type_", &WarpAffineCvtNormTransParam::src_pixel_type_)
      .def_readwrite("dst_pixel_type_", &WarpAffineCvtNormTransParam::dst_pixel_type_)
      .def_readwrite("data_type_", &WarpAffineCvtNormTransParam::data_type_)
      .def_readwrite("data_format_", &WarpAffineCvtNormTransParam::data_format_)
      .def_readwrite("h_", &WarpAffineCvtNormTransParam::h_)
      .def_readwrite("w_", &WarpAffineCvtNormTransParam::w_)
      .def_readwrite("normalize_", &WarpAffineCvtNormTransParam::normalize_)
      .def_property(
          "scale_",
          [](const WarpAffineCvtNormTransParam& self) {
            return py::array_t<float>({4},              // shape
                                      {sizeof(float)},  // strides
                                      self.scale_,      // data pointer
                                      py::cast(self)    // parent object
            );
          },
          [](WarpAffineCvtNormTransParam& self, py::array_t<float> arr) {
            auto buf = arr.request();
            if (buf.ndim != 1 || buf.shape[0] > 4) {
              throw std::runtime_error(
                  "Input array must be 1D with 4 elements");
            }
            float* ptr = static_cast<float*>(buf.ptr);
            for (size_t i = 0; i < buf.shape[0]; i++) {
              self.scale_[i] = ptr[i];
            }
          })
      .def_property(
          "mean_",
          [](const WarpAffineCvtNormTransParam& self) {
            return py::array_t<float>({4},              // shape
                                      {sizeof(float)},  // strides
                                      self.mean_,       // data pointer
                                      py::cast(self)    // parent object
            );
          },
          [](WarpAffineCvtNormTransParam& self, py::array_t<float> arr) {
            auto buf = arr.request();
            if (buf.ndim != 1 || buf.shape[0] > 4) {
              throw std::runtime_error(
                  "Input array must be 1D with 4 elements");
            }
            float* ptr = static_cast<float*>(buf.ptr);
            for (size_t i = 0; i < buf.shape[0]; i++) {
              self.mean_[i] = ptr[i];
            }
          })
      .def_property(
          "std_",
          [](const WarpAffineCvtNormTransParam& self) {
            return py::array_t<float>({4},              // shape
                                      {sizeof(float)},  // strides
                                      self.std_,        // data pointer
                                      py::cast(self)    // parent object
            );
          },
          [](WarpAffineCvtNormTransParam& self, py::array_t<float> arr) {
            auto buf = arr.request();
            if (buf.ndim != 1 || buf.shape[0] > 4) {
              throw std::runtime_error(
                  "Input array must be 1D with 4 elements");
            }
            float* ptr = static_cast<float*>(buf.ptr);
            for (size_t i = 0; i < buf.shape[0]; i++) {
              self.std_[i] = ptr[i];
            }
          })
      .def_readwrite("const_value_", &WarpAffineCvtNormTransParam::const_value_)
      .def_readwrite("interp_type_", &WarpAffineCvtNormTransParam::interp_type_)
      .def_readwrite("border_type_", &WarpAffineCvtNormTransParam::border_type_)
      .def_property(
          "border_val_",
          [](const WarpAffineCvtNormTransParam& self) {
            return py::array_t<double>({4},                    // shape
                                       {sizeof(double)},       // strides
                                       self.border_val_.val_,  // data pointer
                                       py::cast(self)          // parent object
            );
          },
          [](WarpAffineCvtNormTransParam& self, py::array_t<double> arr) {
            auto buf = arr.request();
            if (buf.ndim != 1 || buf.shape[0] > 4) {
              throw std::runtime_error(
                  "Input array must be 1D with 4 elements");
            }
            double* ptr = static_cast<double*>(buf.ptr);
            for (size_t i = 0; i < buf.shape[0]; i++) {
              self.border_val_.val_[i] = ptr[i];
            }
            for (size_t i = buf.shape[0]; i < 4; i++) {
              self.border_val_.val_[i] = 0.0;
            }
          });

  py::class_<CvtNormTransParam, base::Param, std::shared_ptr<CvtNormTransParam>>(m, "CvtNormTransParam")
      .def(py::init<>())
      .def_readwrite("src_pixel_type_", &CvtNormTransParam::src_pixel_type_)
      .def_readwrite("dst_pixel_type_", &CvtNormTransParam::dst_pixel_type_)
      .def_readwrite("data_type_", &CvtNormTransParam::data_type_)
      .def_readwrite("data_format_", &CvtNormTransParam::data_format_)
      .def_readwrite("normalize_", &CvtNormTransParam::normalize_)
      .def_property(
          "scale_",
          [](const CvtNormTransParam& self) {
            return py::array_t<float>({4}, {sizeof(float)}, self.scale_, py::cast(self));
          },
          [](CvtNormTransParam& self, py::array_t<float> arr) {
            auto buf = arr.request();
            if (buf.ndim != 1 || buf.shape[0] > 4) {
              throw std::runtime_error("Input array must be 1D with 4 elements");
            }
            float* ptr = static_cast<float*>(buf.ptr);
            for (size_t i = 0; i < buf.shape[0]; i++) {
              self.scale_[i] = ptr[i];
            }
          })
      .def_property(
          "mean_",
          [](const CvtNormTransParam& self) {
            return py::array_t<float>({4}, {sizeof(float)}, self.mean_, py::cast(self));
          },
          [](CvtNormTransParam& self, py::array_t<float> arr) {
            auto buf = arr.request();
            if (buf.ndim != 1 || buf.shape[0] > 4) {
              throw std::runtime_error("Input array must be 1D with 4 elements");
            }
            float* ptr = static_cast<float*>(buf.ptr);
            for (size_t i = 0; i < buf.shape[0]; i++) {
              self.mean_[i] = ptr[i];
            }
          })
      .def_property(
          "std_",
          [](const CvtNormTransParam& self) {
            return py::array_t<float>({4}, {sizeof(float)}, self.std_, py::cast(self));
          },
          [](CvtNormTransParam& self, py::array_t<float> arr) {
            auto buf = arr.request();
            if (buf.ndim != 1 || buf.shape[0] > 4) {
              throw std::runtime_error("Input array must be 1D with 4 elements");
            }
            float* ptr = static_cast<float*>(buf.ptr);
            for (size_t i = 0; i < buf.shape[0]; i++) {
              self.std_[i] = ptr[i];
            }
          });

  py::class_<CvtResizeNormTransParam, base::Param, std::shared_ptr<CvtResizeNormTransParam>>(m, "CvtResizeNormTransParam")
      .def(py::init<>())
      .def_readwrite("src_pixel_type_", &CvtResizeNormTransParam::src_pixel_type_)
      .def_readwrite("dst_pixel_type_", &CvtResizeNormTransParam::dst_pixel_type_)
      .def_readwrite("interp_type_", &CvtResizeNormTransParam::interp_type_)
      .def_readwrite("h_", &CvtResizeNormTransParam::h_)
      .def_readwrite("w_", &CvtResizeNormTransParam::w_)
      .def_readwrite("data_type_", &CvtResizeNormTransParam::data_type_)
      .def_readwrite("data_format_", &CvtResizeNormTransParam::data_format_)
      .def_readwrite("normalize_", &CvtResizeNormTransParam::normalize_)
      .def_property(
          "scale_",
          [](const CvtResizeNormTransParam& self) {
            return py::array_t<float>({4}, {sizeof(float)}, self.scale_, py::cast(self));
          },
          [](CvtResizeNormTransParam& self, py::array_t<float> arr) {
            auto buf = arr.request();
            if (buf.ndim != 1 || buf.shape[0] > 4) {
              throw std::runtime_error("Input array must be 1D with 4 elements");
            }
            float* ptr = static_cast<float*>(buf.ptr);
            for (size_t i = 0; i < buf.shape[0]; i++) {
              self.scale_[i] = ptr[i];
            }
          })
      .def_property(
          "mean_",
          [](const CvtResizeNormTransParam& self) {
            return py::array_t<float>({4}, {sizeof(float)}, self.mean_, py::cast(self));
          },
          [](CvtResizeNormTransParam& self, py::array_t<float> arr) {
            auto buf = arr.request();
            if (buf.ndim != 1 || buf.shape[0] > 4) {
              throw std::runtime_error("Input array must be 1D with 4 elements");
            }
            float* ptr = static_cast<float*>(buf.ptr);
            for (size_t i = 0; i < buf.shape[0]; i++) {
              self.mean_[i] = ptr[i];
            }
          })
      .def_property(
          "std_",
          [](const CvtResizeNormTransParam& self) {
            return py::array_t<float>({4}, {sizeof(float)}, self.std_, py::cast(self));
          },
          [](CvtResizeNormTransParam& self, py::array_t<float> arr) {
            auto buf = arr.request();
            if (buf.ndim != 1 || buf.shape[0] > 4) {
              throw std::runtime_error("Input array must be 1D with 4 elements");
            }
            float* ptr = static_cast<float*>(buf.ptr);
            for (size_t i = 0; i < buf.shape[0]; i++) {
              self.std_[i] = ptr[i];
            }
          });

  py::class_<CvtResizePadNormTransParam, base::Param, std::shared_ptr<CvtResizePadNormTransParam>>(m, "CvtResizePadNormTransParam")
      .def(py::init<>())
      .def_readwrite("src_pixel_type_", &CvtResizePadNormTransParam::src_pixel_type_)
      .def_readwrite("dst_pixel_type_", &CvtResizePadNormTransParam::dst_pixel_type_)
      .def_readwrite("interp_type_", &CvtResizePadNormTransParam::interp_type_)
      .def_readwrite("data_type_", &CvtResizePadNormTransParam::data_type_)
      .def_readwrite("data_format_", &CvtResizePadNormTransParam::data_format_)
      .def_readwrite("h_", &CvtResizePadNormTransParam::h_)
      .def_readwrite("w_", &CvtResizePadNormTransParam::w_)
      .def_readwrite("normalize_", &CvtResizePadNormTransParam::normalize_)
      .def_readwrite("border_type_", &CvtResizePadNormTransParam::border_type_)
      .def_readwrite("top_", &CvtResizePadNormTransParam::top_)
      .def_readwrite("bottom_", &CvtResizePadNormTransParam::bottom_)
      .def_readwrite("left_", &CvtResizePadNormTransParam::left_)
      .def_readwrite("right_", &CvtResizePadNormTransParam::right_)
      .def_property(
          "scale_",
          [](const CvtResizePadNormTransParam& self) {
            return py::array_t<float>({4}, {sizeof(float)}, self.scale_, py::cast(self));
          },
          [](CvtResizePadNormTransParam& self, py::array_t<float> arr) {
            auto buf = arr.request();
            if (buf.ndim != 1 || buf.shape[0] > 4) {
              throw std::runtime_error("Input array must be 1D with 4 elements");
            }
            float* ptr = static_cast<float*>(buf.ptr);
            for (size_t i = 0; i < buf.shape[0]; i++) {
              self.scale_[i] = ptr[i];
            }
          })
      .def_property(
          "mean_",
          [](const CvtResizePadNormTransParam& self) {
            return py::array_t<float>({4}, {sizeof(float)}, self.mean_, py::cast(self));
          },
          [](CvtResizePadNormTransParam& self, py::array_t<float> arr) {
            auto buf = arr.request();
            if (buf.ndim != 1 || buf.shape[0] > 4) {
              throw std::runtime_error("Input array must be 1D with 4 elements");
            }
            float* ptr = static_cast<float*>(buf.ptr);
            for (size_t i = 0; i < buf.shape[0]; i++) {
              self.mean_[i] = ptr[i];
            }
          })
      .def_property(
          "std_",
          [](const CvtResizePadNormTransParam& self) {
            return py::array_t<float>({4}, {sizeof(float)}, self.std_, py::cast(self));
          },
          [](CvtResizePadNormTransParam& self, py::array_t<float> arr) {
            auto buf = arr.request();
            if (buf.ndim != 1 || buf.shape[0] > 4) {
              throw std::runtime_error("Input array must be 1D with 4 elements");
            }
            float* ptr = static_cast<float*>(buf.ptr);
            for (size_t i = 0; i < buf.shape[0]; i++) {
              self.std_[i] = ptr[i];
            }
          })
      .def_property(
          "border_val_",
          [](const CvtResizePadNormTransParam& self) {
            return py::array_t<double>({4}, {sizeof(double)}, self.border_val_.val_, py::cast(self));
          },
          [](CvtResizePadNormTransParam& self, py::array_t<double> arr) {
            auto buf = arr.request();
            if (buf.ndim != 1 || buf.shape[0] > 4) {
              throw std::runtime_error("Input array must be 1D with 4 elements");
            }
            double* ptr = static_cast<double*>(buf.ptr);
            for (size_t i = 0; i < buf.shape[0]; i++) {
              self.border_val_.val_[i] = ptr[i];
            }
          });

  py::class_<CvtResizeNormTransCropNormTransParam, base::Param, std::shared_ptr<CvtResizeNormTransCropNormTransParam>>(m, "CvtResizeNormTransCropNormTransParam")
      .def(py::init<>())
      .def_readwrite("src_pixel_type_", &CvtResizeNormTransCropNormTransParam::src_pixel_type_)
      .def_readwrite("dst_pixel_type_", &CvtResizeNormTransCropNormTransParam::dst_pixel_type_)
      .def_readwrite("interp_type_", &CvtResizeNormTransCropNormTransParam::interp_type_)
      .def_readwrite("data_type_", &CvtResizeNormTransCropNormTransParam::data_type_)
      .def_readwrite("data_format_", &CvtResizeNormTransCropNormTransParam::data_format_)
      .def_readwrite("resize_h_", &CvtResizeNormTransCropNormTransParam::resize_h_)
      .def_readwrite("resize_w_", &CvtResizeNormTransCropNormTransParam::resize_w_)
      .def_readwrite("normalize_", &CvtResizeNormTransCropNormTransParam::normalize_)
      .def_readwrite("top_left_x_", &CvtResizeNormTransCropNormTransParam::top_left_x_)
      .def_readwrite("top_left_y_", &CvtResizeNormTransCropNormTransParam::top_left_y_)
      .def_readwrite("width_", &CvtResizeNormTransCropNormTransParam::width_)
      .def_readwrite("height_", &CvtResizeNormTransCropNormTransParam::height_)
      .def_property(
          "scale_",
          [](const CvtResizeNormTransCropNormTransParam& self) {
            return py::array_t<float>({4}, {sizeof(float)}, self.scale_, py::cast(self));
          },
          [](CvtResizeNormTransCropNormTransParam& self, py::array_t<float> arr) {
            auto buf = arr.request();
            if (buf.ndim != 1 || buf.shape[0] > 4) {
              throw std::runtime_error("Input array must be 1D with 4 elements");
            }
            float* ptr = static_cast<float*>(buf.ptr);
            for (size_t i = 0; i < buf.shape[0]; i++) {
              self.scale_[i] = ptr[i];
            }
          })
      .def_property(
          "mean_",
          [](const CvtResizeNormTransCropNormTransParam& self) {
            return py::array_t<float>({4}, {sizeof(float)}, self.mean_, py::cast(self));
          },
          [](CvtResizeNormTransCropNormTransParam& self, py::array_t<float> arr) {
            auto buf = arr.request();
            if (buf.ndim != 1 || buf.shape[0] > 4) {
              throw std::runtime_error("Input array must be 1D with 4 elements");
            }
            float* ptr = static_cast<float*>(buf.ptr);
            for (size_t i = 0; i < buf.shape[0]; i++) {
              self.mean_[i] = ptr[i];
            }
          })
      .def_property(
          "std_",
          [](const CvtResizeNormTransCropNormTransParam& self) {
            return py::array_t<float>({4}, {sizeof(float)}, self.std_, py::cast(self));
          },
          [](CvtResizeNormTransCropNormTransParam& self, py::array_t<float> arr) {
            auto buf = arr.request();
            if (buf.ndim != 1 || buf.shape[0] > 4) {
              throw std::runtime_error("Input array must be 1D with 4 elements");
            }
            float* ptr = static_cast<float*>(buf.ptr);
            for (size_t i = 0; i < buf.shape[0]; i++) {
              self.std_[i] = ptr[i];
            }
          });
}

}  // namespace preprocess
}  // namespace nndeploy