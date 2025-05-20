
#include <google/protobuf/message.h>
#include <google/protobuf/text_format.h>

#include "google/protobuf/io/coded_stream.h"
#include "google/protobuf/io/zero_copy_stream_impl.h"
#include "nndeploy/ir/ir.h"
#include "nndeploy/ir/onnx/onnx_interpret.h"

namespace nndeploy {
namespace ir {

class OnnxCastConvert : public OnnxOpConvert {
 public:
  OnnxCastConvert() : OnnxOpConvert() {}
  virtual ~OnnxCastConvert() {}

  virtual std::shared_ptr<OpDesc> convert(const onnx::NodeProto &onnx_node) {
    std::shared_ptr<OpDesc> op_desc = std::make_shared<OpDesc>(kOpTypeCast);
    OnnxOpConvert::convert(onnx_node, op_desc);
    CastParam *param = (CastParam *)(op_desc->op_param_.get());
    param->saturate_ = OnnxInterpret::getAttributeInt(onnx_node, "saturate", 1);

    param->to_ = OnnxInterpret::convertToDataType(
        OnnxInterpret::getAttributeTensorProtoDatatype(onnx_node, "to"));
    return op_desc;
  };
};

REGISTER_ONNX_OP_CONVERT_IMPLEMENTION("Cast", OnnxCastConvert);

}  // namespace ir
}  // namespace nndeploy