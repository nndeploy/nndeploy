
#include <google/protobuf/message.h>
#include <google/protobuf/text_format.h>

#include "google/protobuf/io/coded_stream.h"
#include "google/protobuf/io/zero_copy_stream_impl.h"
#include "nndeploy/ir/ir.h"
#include "nndeploy/ir/onnx/onnx_interpret.h"

namespace nndeploy {
namespace ir {

class OnnxDequantizeLinearConvert : public OnnxOpConvert {
 public:
  OnnxDequantizeLinearConvert() : OnnxOpConvert() {}
  virtual ~OnnxDequantizeLinearConvert() {}

  virtual std::shared_ptr<OpDesc> convert(const onnx::NodeProto &onnx_node) {
    std::shared_ptr<OpDesc> op_desc =
        std::make_shared<OpDesc>(kOpTypeDequantizeLinear);
    OnnxOpConvert::convert(onnx_node, op_desc);
    DequantizeLinearParam *param =
        (DequantizeLinearParam *)(op_desc->op_param_.get());
    param->axis_ = OnnxInterpret::getAttributeInt(onnx_node, "axis", 1);
    return op_desc;
  };
};

REGISTER_ONNX_OP_CONVERT_IMPLEMENTION("DequantizeLinear",
                                      OnnxDequantizeLinearConvert);

}  // namespace ir
}  // namespace nndeploy