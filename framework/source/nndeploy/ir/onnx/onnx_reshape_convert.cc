
#include <google/protobuf/message.h>
#include <google/protobuf/text_format.h>

#include "google/protobuf/io/coded_stream.h"
#include "google/protobuf/io/zero_copy_stream_impl.h"
#include "nndeploy/ir/ir.h"
#include "nndeploy/ir/onnx/onnx_interpret.h"

namespace nndeploy {
namespace ir {

class OnnxReshapeConvert : public OnnxOpConvert {
 public:
  OnnxReshapeConvert() : OnnxOpConvert() {}
  virtual ~OnnxReshapeConvert() {}

  virtual std::shared_ptr<OpDesc> convert(const onnx::NodeProto &onnx_node) {
    std::shared_ptr<OpDesc> op_desc = std::make_shared<OpDesc>(kOpTypeReshape);
    OnnxOpConvert::convert(onnx_node, op_desc);
    ReshapeParam *param = (ReshapeParam *)(op_desc->op_param_.get());
    param->allowzero_ =
        OnnxInterpret::getAttributeInt(onnx_node, "allowzero", 0);
    return op_desc;
  };
};

REGISTER_ONNX_OP_CONVERT_IMPLEMENTION("Reshape", OnnxReshapeConvert);

}  // namespace ir
}  // namespace nndeploy