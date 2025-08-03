#include "nndeploy/codec/codec.h"

#include "nndeploy_api_registry.h"

#ifdef ENABLE_NNDEPLOY_OPENCV
#include "nndeploy/codec/opencv/opencv_codec.h"
#endif

namespace py = pybind11;

namespace nndeploy {
namespace codec {

// 包装纯虚类Decode
class PyDecode : public Decode {
 public:
  using Decode::Decode;  // 继承构造函数

  base::Status setPath(const std::string &path) {
    PYBIND11_OVERRIDE_PURE_NAME(base::Status, Decode, "set_path", setPath,
                                path);
  }

  base::EdgeUpdateFlag updateInput() override {
    PYBIND11_OVERRIDE_PURE_NAME(base::EdgeUpdateFlag, Decode, "update_input",
                                updateInput);
  }

  base::Status run() override {
    PYBIND11_OVERRIDE_PURE_NAME(base::Status, Decode, "run", run);
  }
};

// 包装纯虚类Encode
class PyEncode : public Encode {
 public:
  using Encode::Encode;  // 继承构造函数

  base::Status setRefPath(const std::string &ref_path) {
    PYBIND11_OVERRIDE_PURE_NAME(base::Status, Encode, "set_ref_path",
                                setRefPath, ref_path);
  }

  base::Status setPath(const std::string &path) {
    PYBIND11_OVERRIDE_PURE_NAME(base::Status, Encode, "set_path", setPath,
                                path);
  }

  base::Status run() override {
    PYBIND11_OVERRIDE_PURE_NAME(base::Status, Encode, "run", run);
  }
};

NNDEPLOY_API_PYBIND11_MODULE("codec", m) {
  py::class_<Decode, PyDecode, dag::Node>(m, "Decode")
      .def(py::init<const std::string &, base::CodecFlag>())
      .def(py::init<const std::string &, std::vector<dag::Edge *>,
                    std::vector<dag::Edge *>, base::CodecFlag>())
      .def("set_codec_flag", &Decode::setCodecFlag, py::arg("flag"))
      .def("get_codec_flag", &Decode::getCodecFlag)
      .def("set_path", &Decode::setPath, py::arg("path"))
      .def("set_size", &Decode::setSize, py::arg("size"))
      .def("get_size", &Decode::getSize)
      .def("get_fps", &Decode::getFps)
      .def("get_width", &Decode::getWidth)
      .def("get_height", &Decode::getHeight)
      .def("update_input", &Decode::updateInput)
      .def("run", &Decode::run);

  py::class_<Encode, PyEncode, dag::Node>(m, "Encode")
      .def(py::init<const std::string &, base::CodecFlag>())
      .def(py::init<const std::string &, std::vector<dag::Edge *>,
                    std::vector<dag::Edge *>, base::CodecFlag>())
      .def("set_codec_flag", &Encode::setCodecFlag, py::arg("flag"))
      .def("get_codec_flag", &Encode::getCodecFlag)
      .def("set_path", &Encode::setPath, py::arg("path"))
      .def("set_ref_path", &Encode::setRefPath, py::arg("ref_path"))
      .def("set_size", &Encode::setSize, py::arg("size"))
      .def("get_size", &Encode::getSize)
      .def("set_fourcc", &Encode::setFourcc, py::arg("fourcc"))
      .def("set_fps", &Encode::setFps, py::arg("fps"))
      .def("set_width", &Encode::setWidth, py::arg("width"))
      .def("set_height", &Encode::setHeight, py::arg("height"))
      .def("get_index", &Encode::getIndex)
      .def("run", &Encode::run);

  // 导出创建节点的函数
  m.def("create_decode_node", &createDecode, py::arg("type"), py::arg("flag"),
        py::arg("name"), py::arg("output"),
        py::return_value_policy::take_ownership);
  m.def("create_encode_node", &createEncode, py::arg("type"), py::arg("flag"),
        py::arg("name"), py::arg("input"),
        py::return_value_policy::take_ownership);

#ifdef ENABLE_NNDEPLOY_OPENCV
  py::class_<OpenCvImageDecode, Decode>(m, "OpenCvImageDecode")
      .def(py::init<const std::string &>())
      .def(py::init<const std::string &, std::vector<dag::Edge *> &,
                    std::vector<dag::Edge *> &>())
      .def(py::init<const std::string &, base::CodecFlag>())
      .def(py::init<const std::string &, std::vector<dag::Edge *> &,
                    std::vector<dag::Edge *> &, base::CodecFlag>())
      .def("init", &OpenCvImageDecode::init)
      .def("deinit", &OpenCvImageDecode::deinit)
      .def("set_path", &OpenCvImageDecode::setPath, py::arg("path"))
      .def("run", &OpenCvImageDecode::run);

  py::class_<OpenCvImagesDecode, Decode>(m, "OpenCvImagesDecode")
      .def(py::init<const std::string &>())
      .def(py::init<const std::string &, std::vector<dag::Edge *> &,
                    std::vector<dag::Edge *> &>())
      .def(py::init<const std::string &, base::CodecFlag>())
      .def(py::init<const std::string &, std::vector<dag::Edge *> &,
                    std::vector<dag::Edge *> &, base::CodecFlag>())
      .def("init", &OpenCvImagesDecode::init)
      .def("deinit", &OpenCvImagesDecode::deinit)
      .def("set_path", &OpenCvImagesDecode::setPath, py::arg("path"))
      .def("run", &OpenCvImagesDecode::run);

  py::class_<OpenCvVideoDecode, Decode>(m, "OpenCvVideoDecode")
      .def(py::init<const std::string &>())
      .def(py::init<const std::string &, std::vector<dag::Edge *> &,
                    std::vector<dag::Edge *> &>())
      .def(py::init<const std::string &, base::CodecFlag>())
      .def(py::init<const std::string &, std::vector<dag::Edge *> &,
                    std::vector<dag::Edge *> &, base::CodecFlag>())
      .def("init", &OpenCvVideoDecode::init)
      .def("deinit", &OpenCvVideoDecode::deinit)
      .def("set_path", &OpenCvVideoDecode::setPath, py::arg("path"))
      .def("run", &OpenCvVideoDecode::run);

  py::class_<OpenCvCameraDecode, Decode>(m, "OpenCvCameraDecode")
      .def(py::init<const std::string &>())
      .def(py::init<const std::string &, std::vector<dag::Edge *> &,
                    std::vector<dag::Edge *> &>())
      .def(py::init<const std::string &, base::CodecFlag>())
      .def(py::init<const std::string &, std::vector<dag::Edge *> &,
                    std::vector<dag::Edge *> &, base::CodecFlag>())
      .def("init", &OpenCvCameraDecode::init)
      .def("deinit", &OpenCvCameraDecode::deinit)
      .def("set_path", &OpenCvCameraDecode::setPath, py::arg("path"))
      .def("run", &OpenCvCameraDecode::run);

  py::class_<OpenCvImageEncode, Encode>(m, "OpenCvImageEncode")
      .def(py::init<const std::string &>())
      .def(py::init<const std::string &, std::vector<dag::Edge *> &,
                    std::vector<dag::Edge *> &>())
      .def(py::init<const std::string &, base::CodecFlag>())
      .def(py::init<const std::string &, std::vector<dag::Edge *> &,
                    std::vector<dag::Edge *> &, base::CodecFlag>())
      .def("init", &OpenCvImageEncode::init)
      .def("deinit", &OpenCvImageEncode::deinit)
      .def("set_ref_path", &OpenCvImageEncode::setRefPath, py::arg("ref_path"))
      .def("set_path", &OpenCvImageEncode::setPath, py::arg("path"))
      .def("run", &OpenCvImageEncode::run);

  py::class_<OpenCvImagesEncode, Encode>(m, "OpenCvImagesEncode")
      .def(py::init<const std::string &>())
      .def(py::init<const std::string &, std::vector<dag::Edge *> &,
                    std::vector<dag::Edge *> &>())
      .def(py::init<const std::string &, base::CodecFlag>())
      .def(py::init<const std::string &, std::vector<dag::Edge *> &,
                    std::vector<dag::Edge *> &, base::CodecFlag>())
      .def("init", &OpenCvImagesEncode::init)
      .def("deinit", &OpenCvImagesEncode::deinit)
      .def("set_ref_path", &OpenCvImagesEncode::setRefPath, py::arg("ref_path"))
      .def("set_path", &OpenCvImagesEncode::setPath, py::arg("path"))
      .def("run", &OpenCvImagesEncode::run);

  py::class_<OpenCvVideoEncode, Encode>(m, "OpenCvVideoEncode")
      .def(py::init<const std::string &>())
      .def(py::init<const std::string &, std::vector<dag::Edge *> &,
                    std::vector<dag::Edge *> &>())
      .def(py::init<const std::string &, base::CodecFlag>())
      .def(py::init<const std::string &, std::vector<dag::Edge *> &,
                    std::vector<dag::Edge *> &, base::CodecFlag>())
      .def("init", &OpenCvVideoEncode::init)
      .def("deinit", &OpenCvVideoEncode::deinit)
      .def("set_ref_path", &OpenCvVideoEncode::setRefPath, py::arg("ref_path"))
      .def("set_path", &OpenCvVideoEncode::setPath, py::arg("path"))
      .def("run", &OpenCvVideoEncode::run);

  py::class_<OpenCvCameraEncode, Encode>(m, "OpenCvCameraEncode")
      .def(py::init<const std::string &>())
      .def(py::init<const std::string &, std::vector<dag::Edge *> &,
                    std::vector<dag::Edge *> &>())
      .def(py::init<const std::string &, base::CodecFlag>())
      .def(py::init<const std::string &, std::vector<dag::Edge *> &,
                    std::vector<dag::Edge *> &, base::CodecFlag>())
      .def("init", &OpenCvCameraEncode::init)
      .def("deinit", &OpenCvCameraEncode::deinit)
      .def("set_ref_path", &OpenCvCameraEncode::setRefPath, py::arg("ref_path"))
      .def("set_path", &OpenCvCameraEncode::setPath, py::arg("path"))
      .def("run", &OpenCvCameraEncode::run);

  m.def("create_opencv_decode", &createOpenCvDecode, py::arg("flag"),
        py::arg("name"), py::arg("output"));
  m.def("create_opencv_encode", &createOpenCvEncode, py::arg("flag"),
        py::arg("name"), py::arg("input"));
#endif
}

}  // namespace codec
}  // namespace nndeploy
