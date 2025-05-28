#include <google/protobuf/message.h>
#include <google/protobuf/text_format.h>

#include "google/protobuf/io/coded_stream.h"
#include "google/protobuf/io/zero_copy_stream_impl.h"
#include "nndeploy/ir/ir.h"
#include "nndeploy/ir/onnx/onnx_interpret.h"

namespace nndeploy {
namespace ir {

class OnnxShapeConvert : public OnnxOpConvert {
 public:
  OnnxShapeConvert() : OnnxOpConvert() {}
  virtual ~OnnxShapeConvert() {}

  virtual std::shared_ptr<OpDesc> convert(const onnx::NodeProto &onnx_node) {
    std::shared_ptr<OpDesc> op_desc = std::make_shared<OpDesc>(kOpTypeShape);
    OnnxOpConvert::convert(onnx_node, op_desc);
    ShapeParam *param = (ShapeParam *)(op_desc->op_param_.get());

    // 获取start属性
    param->start_ = OnnxInterpret::getAttributeInt(onnx_node, "start", 0);

    // 获取end属性
    param->end_ = OnnxInterpret::getAttributeInt(onnx_node, "end", -1);

    return op_desc;
  };
};

REGISTER_ONNX_OP_CONVERT_IMPLEMENTION("Shape", OnnxShapeConvert);

}  // namespace ir
}  // namespace nndeploy