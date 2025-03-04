#include "nndeploy/codec/codec.h"

#include "nndeploy_api_registry.h"

namespace py = pybind11;

namespace nndeploy {
namespace codec {

// 包装纯虚类DecodeNode
class PyDecodeNode : public DecodeNode {
 public:
  using DecodeNode::DecodeNode;  // 继承构造函数

  base::Status run() override {
    PYBIND11_OVERRIDE_PURE_NAME(base::Status, DecodeNode, "run", run);
  }
};

// 包装纯虚类EncodeNode
class PyEncodeNode : public EncodeNode {
 public:
  using EncodeNode::EncodeNode;  // 继承构造函数

  base::Status run() override {
    PYBIND11_OVERRIDE_PURE_NAME(base::Status, EncodeNode, "run", run);
  }
};

NNDEPLOY_API_PYBIND11_MODULE("codec", m) {
  py::class_<DecodeNode, PyDecodeNode, dag::Node>(
      m, "DecodeNode")
      .def(py::init<const std::string &, base::CodecFlag>())
      .def(py::init<const std::string &, std::vector<dag::Edge *>,
                   std::vector<dag::Edge *>, base::CodecFlag>())
      .def("set_codec_flag", &DecodeNode::setCodecFlag, py::arg("flag"))
      .def("get_codec_flag", &DecodeNode::getCodecFlag)
      .def("set_path", &DecodeNode::setPath, py::arg("path"))
      .def("set_size", &DecodeNode::setSize, py::arg("size"))
      .def("get_size", &DecodeNode::getSize)
      .def("get_fps", &DecodeNode::getFps)
      .def("get_width", &DecodeNode::getWidth)
      .def("get_height", &DecodeNode::getHeight)
      .def("update_input", &DecodeNode::updataInput)
      .def("run", &DecodeNode::run);

  py::class_<EncodeNode, PyEncodeNode, dag::Node>(
      m, "EncodeNode")
      .def(py::init<const std::string &, base::CodecFlag>())
      .def(py::init<const std::string &, std::vector<dag::Edge *>,
                   std::vector<dag::Edge *>, base::CodecFlag>())
      .def("set_codec_flag", &EncodeNode::setCodecFlag, py::arg("flag"))
      .def("get_codec_flag", &EncodeNode::getCodecFlag)
      .def("set_path", &EncodeNode::setPath, py::arg("path"))
      .def("set_ref_path", &EncodeNode::setRefPath, py::arg("ref_path"))
      .def("set_fourcc", &EncodeNode::setFourcc, py::arg("fourcc"))
      .def("set_fps", &EncodeNode::setFps, py::arg("fps"))
      .def("set_width", &EncodeNode::setWidth, py::arg("width"))
      .def("set_height", &EncodeNode::setHeight, py::arg("height"))
      .def("get_index", &EncodeNode::getIndex)
      .def("run", &EncodeNode::run);

  // 导出创建节点的函数
  m.def("create_decode_node", &createDecodeNode, py::arg("type"),
        py::arg("flag"), py::arg("name"), py::arg("output"), 
        py::return_value_policy::reference);
  // m.def("create_decode_node_shared_ptr", &createDecodeNodeSharedPtr, py::arg("type"),
  //       py::arg("flag"), py::arg("name"), py::arg("output"), 
  //       py::return_value_policy::reference);
  m.def("create_encode_node", &createEncodeNode, py::arg("type"),
        py::arg("flag"), py::arg("name"), py::arg("input"), 
        py::return_value_policy::reference);
  // m.def("create_encode_node_shared_ptr", &createEncodeNodeSharedPtr, py::arg("type"),
  //       py::arg("flag"), py::arg("name"), py::arg("input"), 
  //       py::return_value_policy::reference);
}

}  // namespace codec
}  // namespace nndeploy
