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

  py::class_<OpenCvImagesDecode, Decode>(m, "OpenCvImagesDecode")
      .def(py::init<const std::string &>())
      .def(py::init<const std::string &, std::vector<dag::Edge *> &, std::vector<dag::Edge *> &>())
      .def(py::init<const std::string &, base::CodecFlag>())
      .def(py::init<const std::string &, std::vector<dag::Edge *> &, std::vector<dag::Edge *> &, base::CodecFlag>())
      .def("init", &OpenCvImagesDecode::init)
      .def("deinit", &OpenCvImagesDecode::deinit)
      .def("set_path", &OpenCvImagesDecode::setPath, py::arg("path"))
      .def("run", &OpenCvImagesDecode::run);

  py::class_<OpenCvVedioDecode, Decode>(m, "OpenCvVedioDecode")
      .def(py::init<const std::string &>())
      .def(py::init<const std::string &, std::vector<dag::Edge *> &, std::vector<dag::Edge *> &>())
      .def(py::init<const std::string &, base::CodecFlag>())
      .def(py::init<const std::string &, std::vector<dag::Edge *> &, std::vector<dag::Edge *> &, base::CodecFlag>())
      .def("init", &OpenCvVedioDecode::init)
      .def("deinit", &OpenCvVedioDecode::deinit)
      .def("set_path", &OpenCvVedioDecode::setPath, py::arg("path"))
      .def("run", &OpenCvVedioDecode::run);

  py::class_<OpenCvCameraDecode, Decode>(m, "OpenCvCameraDecode")
      .def(py::init<const std::string &>())
      .def(py::init<const std::string &, std::vector<dag::Edge *> &, std::vector<dag::Edge *> &>())
      .def(py::init<const std::string &, base::CodecFlag>())
      .def(py::init<const std::string &, std::vector<dag::Edge *> &, std::vector<dag::Edge *> &, base::CodecFlag>())
      .def("init", &OpenCvCameraDecode::init)
      .def("deinit", &OpenCvCameraDecode::deinit)
      .def("set_path", &OpenCvCameraDecode::setPath, py::arg("path"))
      .def("run", &OpenCvCameraDecode::run);

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

  py::class_<OpenCvImagesEncode, Encode>(m, "OpenCvImagesEncode")
      .def(py::init<const std::string &>())
      .def(py::init<const std::string &, std::vector<dag::Edge *> &, std::vector<dag::Edge *> &>())
      .def(py::init<const std::string &, base::CodecFlag>())
      .def(py::init<const std::string &, std::vector<dag::Edge *> &, std::vector<dag::Edge *> &, base::CodecFlag>())
      .def("init", &OpenCvImagesEncode::init)
      .def("deinit", &OpenCvImagesEncode::deinit)
      .def("set_ref_path", &OpenCvImagesEncode::setRefPath, py::arg("ref_path"))
      .def("set_path", &OpenCvImagesEncode::setPath, py::arg("path"))
      .def("run", &OpenCvImagesEncode::run);

  py::class_<OpenCvVedioEncode, Encode>(m, "OpenCvVedioEncode")
      .def(py::init<const std::string &>())
      .def(py::init<const std::string &, std::vector<dag::Edge *> &, std::vector<dag::Edge *> &>())
      .def(py::init<const std::string &, base::CodecFlag>())
      .def(py::init<const std::string &, std::vector<dag::Edge *> &, std::vector<dag::Edge *> &, base::CodecFlag>())
      .def("init", &OpenCvVedioEncode::init)
      .def("deinit", &OpenCvVedioEncode::deinit)
      .def("set_ref_path", &OpenCvVedioEncode::setRefPath, py::arg("ref_path"))
      .def("set_path", &OpenCvVedioEncode::setPath, py::arg("path"))
      .def("run", &OpenCvVedioEncode::run);

  py::class_<OpenCvCameraEncode, Encode>(m, "OpenCvCameraEncode")
      .def(py::init<const std::string &>())
      .def(py::init<const std::string &, std::vector<dag::Edge *> &, std::vector<dag::Edge *> &>())
      .def(py::init<const std::string &, base::CodecFlag>())
      .def(py::init<const std::string &, std::vector<dag::Edge *> &, std::vector<dag::Edge *> &, base::CodecFlag>())
      .def("init", &OpenCvCameraEncode::init)
      .def("deinit", &OpenCvCameraEncode::deinit)
      .def("set_ref_path", &OpenCvCameraEncode::setRefPath, py::arg("ref_path"))
      .def("set_path", &OpenCvCameraEncode::setPath, py::arg("path"))
      .def("run", &OpenCvCameraEncode::run);

  m.def("create_opencv_decode", &createOpenCvDecode, py::arg("flag"), py::arg("name"), py::arg("output"));
  m.def("create_opencv_encode", &createOpenCvEncode, py::arg("flag"), py::arg("name"), py::arg("input"));
}

}  // namespace codec
}  // namespace nndeploy
