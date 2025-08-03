
#include <google/protobuf/message.h>
#include <google/protobuf/text_format.h>

#include "google/protobuf/io/coded_stream.h"
#include "google/protobuf/io/zero_copy_stream_impl.h"
#include "nndeploy/ir/ir.h"
#include "nndeploy/ir/onnx/onnx_interpret.h"

namespace nndeploy {
namespace ir {

class OnnxConstantOfShapeConvert : public OnnxOpConvert {
 public:
  OnnxConstantOfShapeConvert() : OnnxOpConvert() {}
  virtual ~OnnxConstantOfShapeConvert() {}

  virtual std::shared_ptr<OpDesc> convert(const onnx::NodeProto &onnx_node) {
    std::shared_ptr<OpDesc> op_desc =
        std::make_shared<OpDesc>(kOpTypeConstantOfShape);
    OnnxOpConvert::convert(onnx_node, op_desc);
    ConstantOfShapeParam* param =
        (ConstantOfShapeParam*)(op_desc->op_param_.get());
    onnx::TensorProto onnx_value =
        OnnxInterpret::getAttributeTensor(onnx_node, "value");
    // value 在onnx文档中描述为可选的
    device::Tensor* value = OnnxInterpret::convertToTensor(onnx_value);

    return op_desc;
  };
};

REGISTER_ONNX_OP_CONVERT_IMPLEMENTION("ConstantOfShape",
                                      OnnxConstantOfShapeConvert);

}  // namespace ir
}  // namespace nndeploy