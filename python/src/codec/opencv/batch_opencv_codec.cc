#include "nndeploy/codec/opencv/batch_opencv_codec.h"

#include "nndeploy/base/file.h"
#include "nndeploy_api_registry.h"

namespace py = pybind11;

namespace nndeploy {
namespace codec {

template <typename Base = BatchOpenCvDecode>
class PyBatchOpenCvDecode : public Base {
 public:
  using Base::Base;  // 继承构造函数

  // base::Status setNodeKey(const std::string &key) {
  //   PYBIND11_OVERRIDE_NAME(base::Status, BatchOpenCvDecode, "set_node_key",
  //   setNodeKey, key);
  // }

  // base::Status setCodecFlag(base::CodecFlag flag) {
  //   PYBIND11_OVERRIDE_NAME(base::Status, BatchOpenCvDecode, "set_codec_flag",
  //   setCodecFlag, flag);
  // }

  // base::CodecFlag getCodecFlag() {
  //   PYBIND11_OVERRIDE_NAME(base::CodecFlag, BatchOpenCvDecode,
  //   "get_codec_flag", getCodecFlag);
  // }

  base::EdgeUpdateFlag updateInput() {
    PYBIND11_OVERRIDE_NAME(base::EdgeUpdateFlag, BatchOpenCvDecode,
                           "update_input", updateInput);
  }

  base::Status run() {
    PYBIND11_OVERRIDE_NAME(base::Status, BatchOpenCvDecode, "run", run);
  }

  // std::string serialize() {
  //   PYBIND11_OVERRIDE_NAME(std::string, BatchOpenCvDecode, "serialize",
  //                          serialize);
  // }

  // base::Status deserialize(const std::string &json_str) {
  //   PYBIND11_OVERRIDE_NAME(base::Status, BatchOpenCvDecode, "deserialize",
  //                          deserialize, json_str);
  // }
};

template <typename Base = BatchOpenCvEncode>
class PyBatchOpenCvEncode : public Base {
 public:
  using Base::Base;  // 继承构造函数

  // base::Status setNodeKey(const std::string &key) {
  //   PYBIND11_OVERRIDE_NAME(base::Status, BatchOpenCvEncode, "set_node_key",
  //   setNodeKey, key);
  // }

  // base::Status setCodecFlag(base::CodecFlag flag) {
  //   PYBIND11_OVERRIDE_NAME(base::Status, BatchOpenCvEncode, "set_codec_flag",
  //   setCodecFlag, flag);
  // }

  // base::CodecFlag getCodecFlag() {
  //   PYBIND11_OVERRIDE_NAME(base::CodecFlag, BatchOpenCvEncode,
  //   "get_codec_flag", getCodecFlag);
  // }

  base::Status run() {
    PYBIND11_OVERRIDE_NAME(base::Status, BatchOpenCvEncode, "run", run);
  }

  // std::string serialize() {
  //   PYBIND11_OVERRIDE_NAME(std::string, BatchOpenCvEncode, "serialize",
  //                          serialize);
  // }

  // base::Status deserialize(const std::string &json_str) {
  //   PYBIND11_OVERRIDE_NAME(base::Status, BatchOpenCvEncode, "deserialize",
  //                          deserialize, json_str);
  // }
};

NNDEPLOY_API_PYBIND11_MODULE("codec", m) {
  py::class_<BatchOpenCvDecode, PyBatchOpenCvDecode<BatchOpenCvDecode>,
             dag::Node>(m, "BatchOpenCvDecode")
      .def(py::init<const std::string &>())
      .def("setBatchSize", &BatchOpenCvDecode::setBatchSize)
      .def("setNodeKey", &BatchOpenCvDecode::setNodeKey)
      .def("setCodecFlag", &BatchOpenCvDecode::setCodecFlag)
      .def("getCodecFlag", &BatchOpenCvDecode::getCodecFlag)
      .def("setPath", &BatchOpenCvDecode::setPath)
      .def("setSize", &BatchOpenCvDecode::setSize)
      .def("getSize", &BatchOpenCvDecode::getSize)
      .def("getFps", &BatchOpenCvDecode::getFps)
      .def("getWidth", &BatchOpenCvDecode::getWidth)
      .def("getHeight", &BatchOpenCvDecode::getHeight)
      .def("updateInput", &BatchOpenCvDecode::updateInput)
      .def("run", &BatchOpenCvDecode::run);
      // .def("serialize",
      //      py::overload_cast<rapidjson::Value &,
      //                        rapidjson::Document::AllocatorType &>(
      //          &BatchOpenCvDecode::serialize),
      //      py::arg("json"), py::arg("allocator"))
      // .def("serialize", py::overload_cast<>(&BatchOpenCvDecode::serialize))
      // .def("deserialize",
      //      py::overload_cast<rapidjson::Value &>(
      //          &BatchOpenCvDecode::deserialize),
      //      py::arg("json"))
      // .def("deserialize",
      //      py::overload_cast<const std::string &>(
      //          &BatchOpenCvDecode::deserialize),
      //      py::arg("json_str"));

  py::class_<BatchOpenCvEncode, PyBatchOpenCvEncode<BatchOpenCvEncode>,
             dag::Node>(m, "BatchOpenCvEncode")
      .def(py::init<const std::string &>())
      .def("setNodeKey", &BatchOpenCvEncode::setNodeKey)
      .def("setCodecFlag", &BatchOpenCvEncode::setCodecFlag)
      .def("getCodecFlag", &BatchOpenCvEncode::getCodecFlag)
      .def("setPath", &BatchOpenCvEncode::setPath)
      .def("setRefPath", &BatchOpenCvEncode::setRefPath)
      .def("setFourcc", &BatchOpenCvEncode::setFourcc)
      .def("setFps", &BatchOpenCvEncode::setFps)
      .def("setWidth", &BatchOpenCvEncode::setWidth)
      .def("setHeight", &BatchOpenCvEncode::setHeight)
      .def("setSize", &BatchOpenCvEncode::setSize)
      .def("getSize", &BatchOpenCvEncode::getSize)
      .def("getIndex", &BatchOpenCvEncode::getIndex)
      .def("run", &BatchOpenCvEncode::run);
      // .def("serialize",
      //      py::overload_cast<rapidjson::Value &,
      //                        rapidjson::Document::AllocatorType &>(
      //          &BatchOpenCvEncode::serialize),
      //      py::arg("json"), py::arg("allocator"))
      // .def("serialize", py::overload_cast<>(&BatchOpenCvEncode::serialize))
      // .def("deserialize",
      //      py::overload_cast<rapidjson::Value &>(
      //          &BatchOpenCvEncode::deserialize),
      //      py::arg("json"))
      // .def("deserialize",
      //      py::overload_cast<const std::string &>(
      //          &BatchOpenCvEncode::deserialize),
      //      py::arg("json_str"));
}

}  // namespace codec
}  // namespace nndeploy
