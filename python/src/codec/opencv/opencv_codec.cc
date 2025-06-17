#include "nndeploy/codec/opencv/opencv_codec.h"

#include "nndeploy/base/file.h"
#include "nndeploy_api_registry.h"

namespace py = pybind11;

namespace nndeploy {
namespace codec {

NNDEPLOY_API_PYBIND11_MODULE("codec", m) {
  py::class_<OpenCvImageDecode, Decode>(m, "OpenCvImageDecode")
      .def(py::init<const std::string &>())
      .def(py::init<const std::string &, std::vector<dag::Edge *> &, std::vector<dag::Edge *> &>())
      .def(py::init<const std::string &, base::CodecFlag>())
      .def(py::init<const std::string &, std::vector<dag::Edge *> &, std::vector<dag::Edge *> &, base::CodecFlag>())
      .def("init", &OpenCvImageDecode::init)
      .def("deinit", &OpenCvImageDecode::deinit)
      .def("set_path", &OpenCvImageDecode::setPath, py::arg("path"))
      .def("run", &OpenCvImageDecode::run);

  py::class_<OpenCvImageEncode, Encode>(m, "OpenCvImageEncode")
      .def(py::init<const std::string &>())
      .def(py::init<const std::string &, std::vector<dag::Edge *> &, std::vector<dag::Edge *> &>())
      .def(py::init<const std::string &, base::CodecFlag>())
      .def(py::init<const std::string &, std::vector<dag::Edge *> &, std::vector<dag::Edge *> &, base::CodecFlag>())
      .def("init", &OpenCvImageEncode::init)
      .def("deinit", &OpenCvImageEncode::deinit)
      .def("set_ref_path", &OpenCvImageEncode::setRefPath, py::arg("ref_path"))
      .def("set_path", &OpenCvImageEncode::setPath, py::arg("path"))
      .def("run", &OpenCvImageEncode::run);

}

}  // namespace codec
}  // namespace nndeploy
