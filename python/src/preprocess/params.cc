
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

  py::class_<NomalizeParam, base::Param, std::shared_ptr<NomalizeParam>>(m, "NomalizeParam")
      .def(py::init<>())
      .def_property(
          "scale_",
          [](const NomalizeParam& self) {
            return py::array_t<float>({4},              // shape
                                      {sizeof(float)},  // strides
                                      self.scale_,      // data pointer
                                      py::cast(self)    // parent object
            );
          },
          [](NomalizeParam& self, py::array_t<float> arr) {
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
          [](const NomalizeParam& self) {
            return py::array_t<float>({4},              // shape
                                      {sizeof(float)},  // strides
                                      self.mean_,       // data pointer
                                      py::cast(self)    // parent object
            );
          },
          [](NomalizeParam& self, py::array_t<float> arr) {
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
          [](const NomalizeParam& self) {
            return py::array_t<float>({4},              // shape
                                      {sizeof(float)},  // strides
                                      self.std_,        // data pointer
                                      py::cast(self)    // parent object
            );
          },
          [](NomalizeParam& self, py::array_t<float> arr) {
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

  py::class_<WarpAffineParam, base::Param, std::shared_ptr<WarpAffineParam>>(
      m, "WarpAffineParam")
      .def(py::init<>())
      .def_property(
          "transform_",
          [](const WarpAffineParam& self) {
            return py::array_t<float>(
                std::vector<ssize_t>{2, 3},  // shape
                std::vector<ssize_t>{sizeof(float) * 3,
                                     sizeof(float)},  // strides
                &self.transform_[0][0],               // data pointer
                py::cast(self)                        // parent object
            );
          },
          [](WarpAffineParam& self, py::array_t<float> arr) {
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
      .def_readwrite("dst_w_", &WarpAffineParam::dst_w_)
      .def_readwrite("dst_h_", &WarpAffineParam::dst_h_)
      .def_readwrite("src_pixel_type_", &WarpAffineParam::src_pixel_type_)
      .def_readwrite("dst_pixel_type_", &WarpAffineParam::dst_pixel_type_)
      .def_readwrite("data_type_", &WarpAffineParam::data_type_)
      .def_readwrite("data_format_", &WarpAffineParam::data_format_)
      .def_readwrite("h_", &WarpAffineParam::h_)
      .def_readwrite("w_", &WarpAffineParam::w_)
      .def_readwrite("normalize_", &WarpAffineParam::normalize_)
      .def_property(
          "scale_",
          [](const WarpAffineParam& self) {
            return py::array_t<float>({4},              // shape
                                      {sizeof(float)},  // strides
                                      self.scale_,      // data pointer
                                      py::cast(self)    // parent object
            );
          },
          [](WarpAffineParam& self, py::array_t<float> arr) {
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
          [](const WarpAffineParam& self) {
            return py::array_t<float>({4},              // shape
                                      {sizeof(float)},  // strides
                                      self.mean_,       // data pointer
                                      py::cast(self)    // parent object
            );
          },
          [](WarpAffineParam& self, py::array_t<float> arr) {
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
          [](const WarpAffineParam& self) {
            return py::array_t<float>({4},              // shape
                                      {sizeof(float)},  // strides
                                      self.std_,        // data pointer
                                      py::cast(self)    // parent object
            );
          },
          [](WarpAffineParam& self, py::array_t<float> arr) {
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
      .def_readwrite("const_value_", &WarpAffineParam::const_value_)
      .def_readwrite("interp_type_", &WarpAffineParam::interp_type_)
      .def_readwrite("border_type_", &WarpAffineParam::border_type_)
      .def_property(
          "border_val_",
          [](const WarpAffineParam& self) {
            return py::array_t<double>({4},                    // shape
                                       {sizeof(double)},       // strides
                                       self.border_val_.val_,  // data pointer
                                       py::cast(self)          // parent object
            );
          },
          [](WarpAffineParam& self, py::array_t<double> arr) {
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

  py::class_<CvtclorResizeParam, base::Param, std::shared_ptr<CvtclorResizeParam>>(m, "CvtclorResizeParam")
      .def(py::init<>())
      .def_readwrite("src_pixel_type_", &CvtclorResizeParam::src_pixel_type_)
      .def_readwrite("dst_pixel_type_", &CvtclorResizeParam::dst_pixel_type_)
      .def_readwrite("interp_type_", &CvtclorResizeParam::interp_type_)
      .def_readwrite("h_", &CvtclorResizeParam::h_)
      .def_readwrite("w_", &CvtclorResizeParam::w_)
      .def_readwrite("data_type_", &CvtclorResizeParam::data_type_)
      .def_readwrite("data_format_", &CvtclorResizeParam::data_format_)
      .def_readwrite("normalize_", &CvtclorResizeParam::normalize_)
      .def_property(
          "scale_",
          [](const CvtclorResizeParam& self) {
            return py::array_t<float>({4}, {sizeof(float)}, self.scale_, py::cast(self));
          },
          [](CvtclorResizeParam& self, py::array_t<float> arr) {
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
          [](const CvtclorResizeParam& self) {
            return py::array_t<float>({4}, {sizeof(float)}, self.mean_, py::cast(self));
          },
          [](CvtclorResizeParam& self, py::array_t<float> arr) {
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
          [](const CvtclorResizeParam& self) {
            return py::array_t<float>({4}, {sizeof(float)}, self.std_, py::cast(self));
          },
          [](CvtclorResizeParam& self, py::array_t<float> arr) {
            auto buf = arr.request();
            if (buf.ndim != 1 || buf.shape[0] > 4) {
              throw std::runtime_error("Input array must be 1D with 4 elements");
            }
            float* ptr = static_cast<float*>(buf.ptr);
            for (size_t i = 0; i < buf.shape[0]; i++) {
              self.std_[i] = ptr[i];
            }
          });

  py::class_<CvtclorResizePadParam, base::Param, std::shared_ptr<CvtclorResizePadParam>>(m, "CvtclorResizePadParam")
      .def(py::init<>())
      .def_readwrite("src_pixel_type_", &CvtclorResizePadParam::src_pixel_type_)
      .def_readwrite("dst_pixel_type_", &CvtclorResizePadParam::dst_pixel_type_)
      .def_readwrite("interp_type_", &CvtclorResizePadParam::interp_type_)
      .def_readwrite("data_type_", &CvtclorResizePadParam::data_type_)
      .def_readwrite("data_format_", &CvtclorResizePadParam::data_format_)
      .def_readwrite("h_", &CvtclorResizePadParam::h_)
      .def_readwrite("w_", &CvtclorResizePadParam::w_)
      .def_readwrite("normalize_", &CvtclorResizePadParam::normalize_)
      .def_readwrite("border_type_", &CvtclorResizePadParam::border_type_)
      .def_readwrite("top_", &CvtclorResizePadParam::top_)
      .def_readwrite("bottom_", &CvtclorResizePadParam::bottom_)
      .def_readwrite("left_", &CvtclorResizePadParam::left_)
      .def_readwrite("right_", &CvtclorResizePadParam::right_)
      .def_property(
          "scale_",
          [](const CvtclorResizePadParam& self) {
            return py::array_t<float>({4}, {sizeof(float)}, self.scale_, py::cast(self));
          },
          [](CvtclorResizePadParam& self, py::array_t<float> arr) {
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
          [](const CvtclorResizePadParam& self) {
            return py::array_t<float>({4}, {sizeof(float)}, self.mean_, py::cast(self));
          },
          [](CvtclorResizePadParam& self, py::array_t<float> arr) {
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
          [](const CvtclorResizePadParam& self) {
            return py::array_t<float>({4}, {sizeof(float)}, self.std_, py::cast(self));
          },
          [](CvtclorResizePadParam& self, py::array_t<float> arr) {
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
          [](const CvtclorResizePadParam& self) {
            return py::array_t<double>({4}, {sizeof(double)}, self.border_val_.val_, py::cast(self));
          },
          [](CvtclorResizePadParam& self, py::array_t<double> arr) {
            auto buf = arr.request();
            if (buf.ndim != 1 || buf.shape[0] > 4) {
              throw std::runtime_error("Input array must be 1D with 4 elements");
            }
            double* ptr = static_cast<double*>(buf.ptr);
            for (size_t i = 0; i < buf.shape[0]; i++) {
              self.border_val_.val_[i] = ptr[i];
            }
          });

  py::class_<CvtColorResizeCropParam, base::Param, std::shared_ptr<CvtColorResizeCropParam>>(m, "CvtColorResizeCropParam")
      .def(py::init<>())
      .def_readwrite("src_pixel_type_", &CvtColorResizeCropParam::src_pixel_type_)
      .def_readwrite("dst_pixel_type_", &CvtColorResizeCropParam::dst_pixel_type_)
      .def_readwrite("interp_type_", &CvtColorResizeCropParam::interp_type_)
      .def_readwrite("data_type_", &CvtColorResizeCropParam::data_type_)
      .def_readwrite("data_format_", &CvtColorResizeCropParam::data_format_)
      .def_readwrite("resize_h_", &CvtColorResizeCropParam::resize_h_)
      .def_readwrite("resize_w_", &CvtColorResizeCropParam::resize_w_)
      .def_readwrite("normalize_", &CvtColorResizeCropParam::normalize_)
      .def_readwrite("top_left_x_", &CvtColorResizeCropParam::top_left_x_)
      .def_readwrite("top_left_y_", &CvtColorResizeCropParam::top_left_y_)
      .def_readwrite("width_", &CvtColorResizeCropParam::width_)
      .def_readwrite("height_", &CvtColorResizeCropParam::height_)
      .def_property(
          "scale_",
          [](const CvtColorResizeCropParam& self) {
            return py::array_t<float>({4}, {sizeof(float)}, self.scale_, py::cast(self));
          },
          [](CvtColorResizeCropParam& self, py::array_t<float> arr) {
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
          [](const CvtColorResizeCropParam& self) {
            return py::array_t<float>({4}, {sizeof(float)}, self.mean_, py::cast(self));
          },
          [](CvtColorResizeCropParam& self, py::array_t<float> arr) {
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
          [](const CvtColorResizeCropParam& self) {
            return py::array_t<float>({4}, {sizeof(float)}, self.std_, py::cast(self));
          },
          [](CvtColorResizeCropParam& self, py::array_t<float> arr) {
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