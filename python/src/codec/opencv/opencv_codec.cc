#include "nndeploy/codec/opencv/opencv_codec.h"

#include "nndeploy/base/file.h"
#include "nndeploy_api_registry.h"

namespace py = pybind11;

namespace nndeploy {
namespace codec {

NNDEPLOY_API_PYBIND11_MODULE("codec", m) {
  py::class_<OpenCvImageDecodeNode, DecodeNode>(m, "OpenCvImageDecodeNode")
      .def(py::init<const std::string &>())
      .def(py::init<const std::string &, std::vector<dag::Edge *> &, std::vector<dag::Edge *> &>())
      .def(py::init<const std::string &, base::CodecFlag>())
      .def(py::init<const std::string &, std::vector<dag::Edge *> &, std::vector<dag::Edge *> &, base::CodecFlag>())
      .def("init", &OpenCvImageDecodeNode::init)
      .def("deinit", &OpenCvImageDecodeNode::deinit)
      .def("set_path", &OpenCvImageDecodeNode::setPath, py::arg("path"))
      .def("run", &OpenCvImageDecodeNode::run);

  py::class_<OpenCvImageEncodeNode, EncodeNode>(m, "OpenCvImageEncodeNode")
      .def(py::init<const std::string &>())
      .def(py::init<const std::string &, std::vector<dag::Edge *> &, std::vector<dag::Edge *> &>())
      .def(py::init<const std::string &, base::CodecFlag>())
      .def(py::init<const std::string &, std::vector<dag::Edge *> &, std::vector<dag::Edge *> &, base::CodecFlag>())
      .def("init", &OpenCvImageEncodeNode::init)
      .def("deinit", &OpenCvImageEncodeNode::deinit)
      .def("set_ref_path", &OpenCvImageEncodeNode::setRefPath, py::arg("ref_path"))
      .def("set_path", &OpenCvImageEncodeNode::setPath, py::arg("path"))
      .def("run", &OpenCvImageEncodeNode::run);

}

}  // namespace codec
}  // namespace nndeploy
