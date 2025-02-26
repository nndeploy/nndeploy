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
    PYBIND11_OVERRIDE_PURE(base::Status, DecodeNode, run);
  }
};

// 包装纯虚类EncodeNode  
class PyEncodeNode : public EncodeNode {
public:
  using EncodeNode::EncodeNode;  // 继承构造函数

  base::Status run() override {
    PYBIND11_OVERRIDE_PURE(base::Status, EncodeNode, run);
  }
};

NNDEPLOY_API_PYBIND11_MODULE("codec", m) {
  // 导出DecodeNode类
  py::class_<DecodeNode, PyDecodeNode, dag::Node, std::shared_ptr<DecodeNode>>(m, "DecodeNode")
    .def("get_codec_flag", &DecodeNode::getCodecFlag)
    .def("set_path", &DecodeNode::setPath)
    .def("set_size", &DecodeNode::setSize)
    .def("get_size", &DecodeNode::getSize)
    .def("get_fps", &DecodeNode::getFps)
    .def("get_width", &DecodeNode::getWidth)
    .def("get_height", &DecodeNode::getHeight)
    .def("update_input", &DecodeNode::updateInput)
    .def("run", &DecodeNode::run);

  // 导出EncodeNode类
  py::class_<EncodeNode, PyEncodeNode, dag::Node, std::shared_ptr<EncodeNode>>(m, "EncodeNode")
    .def("get_codec_flag", &EncodeNode::getCodecFlag)
    .def("set_path", &EncodeNode::setPath)
    .def("set_ref_path", &EncodeNode::setRefPath)
    .def("set_fourcc", &EncodeNode::setFourcc)
    .def("set_fps", &EncodeNode::setFps)
    .def("set_width", &EncodeNode::setWidth)
    .def("set_height", &EncodeNode::setHeight)
    .def("get_index", &EncodeNode::getIndex)
    .def("run", &EncodeNode::run);

  // 导出创建节点的函数
  m.def("create_decode_node", &createDecodeNode);
  m.def("create_encode_node", &createEncodeNode);
}

}  // namespace codec
}  // namespace nndeploy

