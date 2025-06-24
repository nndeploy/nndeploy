#include "nndeploy/track/fairmot/fairmot.h"

#include "nndeploy/track/result.h"
#include "nndeploy_api_registry.h"

namespace py = pybind11;

namespace nndeploy {
namespace track {

NNDEPLOY_API_PYBIND11_MODULE("track", m) {
  py::class_<FairMotPreParam, base::Param, std::shared_ptr<FairMotPreParam>>(
      m, "FairMotPreParam")
      .def(py::init<>())
      .def_readwrite("src_pixel_type_", &FairMotPreParam::src_pixel_type_)
      .def_readwrite("dst_pixel_type_", &FairMotPreParam::dst_pixel_type_)
      .def_readwrite("interp_type_", &FairMotPreParam::interp_type_)
      .def_readwrite("h_", &FairMotPreParam::h_)
      .def_readwrite("w_", &FairMotPreParam::w_)
      .def_readwrite("data_type_", &FairMotPreParam::data_type_)
      .def_readwrite("data_format_", &FairMotPreParam::data_format_)
      .def_readwrite("normalize_", &FairMotPreParam::normalize_)
      .def_property(
          "scale_",
          [](const FairMotPreParam &self) {
            return py::array_t<float>({4},              // shape
                                      {sizeof(float)},  // strides
                                      self.scale_,      // data pointer
                                      py::cast(self)    // parent object
            );
          },
          [](FairMotPreParam &self, py::array_t<float> arr) {
            auto buf = arr.request();
            if (buf.ndim != 1 || buf.shape[0] > 4) {
              throw std::runtime_error(
                  "Input array must be 1D with 4 elements");
            }
            float *ptr = static_cast<float *>(buf.ptr);
            for (size_t i = 0; i < buf.shape[0]; i++) {
              self.scale_[i] = ptr[i];
            }
          })
      .def_property(
          "mean_",
          [](const FairMotPreParam &self) {
            return py::array_t<float>({4},              // shape
                                      {sizeof(float)},  // strides
                                      self.mean_,       // data pointer
                                      py::cast(self)    // parent object
            );
          },
          [](FairMotPreParam &self, py::array_t<float> arr) {
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
          [](const FairMotPreParam &self) {
            return py::array_t<float>({4},              // shape
                                      {sizeof(float)},  // strides
                                      self.std_,        // data pointer
                                      py::cast(self)    // parent object
            );
          },
          [](FairMotPreParam &self, py::array_t<float> arr) {
            auto buf = arr.request();
            if (buf.ndim != 1 || buf.shape[0] > 4) {
              throw std::runtime_error(
                  "Input array must be 1D with 4 elements");
            }
            float *ptr = static_cast<float *>(buf.ptr);
            for (size_t i = 0; i < buf.shape[0]; i++) {
              self.std_[i] = ptr[i];
            }
          });
          
  py::class_<FairMotPostParam, base::Param, std::shared_ptr<FairMotPostParam>>(
      m, "FairMotPostParam")
      .def(py::init<>())
      .def_readwrite("conf_thresh_", &FairMotPostParam::conf_thresh_)
      .def_readwrite("tracked_thresh_", &FairMotPostParam::tracked_thresh_)
      .def_readwrite("min_box_area_", &FairMotPostParam::min_box_area_);

  py::class_<FairMotPreProcess, dag::Node>(m, "FairMotPreProcess")
      .def(py::init<const std::string &>())
      .def(py::init<const std::string &, std::vector<dag::Edge *>,
                    std::vector<dag::Edge *>>())
      .def("run", &FairMotPreProcess::run);

  py::class_<FairMotPostProcess, dag::Node>(m, "FairMotPostProcess")
      .def(py::init<const std::string &>())
      .def(py::init<const std::string &, std::vector<dag::Edge *>,
                    std::vector<dag::Edge *>>())
      .def("init", &FairMotPostProcess::init)
      .def("deinit", &FairMotPostProcess::deinit)
      .def("run", &FairMotPostProcess::run);

  py::class_<FairMotGraph, dag::Graph>(m, "FairMotGraph")
      .def(py::init<const std::string &>())
      .def(py::init<const std::string &, std::vector<dag::Edge *>,
                    std::vector<dag::Edge *>>())
      .def("default_param", &FairMotGraph::defaultParam)
      .def("make", &FairMotGraph::make)
      .def("set_infer_param", &FairMotGraph::setInferParam);
}

}  // namespace track
}  // namespace nndeploy
