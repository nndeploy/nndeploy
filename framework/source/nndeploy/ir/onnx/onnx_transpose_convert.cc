
#include <google/protobuf/message.h>
#include <google/protobuf/text_format.h>

#include "google/protobuf/io/coded_stream.h"
#include "google/protobuf/io/zero_copy_stream_impl.h"
#include "nndeploy/ir/ir.h"
#include "nndeploy/ir/onnx/onnx_interpret.h"

namespace nndeploy {
namespace ir {

class OnnxTransposeConvert : public OnnxOpConvert {
 public:
  OnnxTransposeConvert() : OnnxOpConvert() {}
  virtual ~OnnxTransposeConvert() {}

  virtual std::shared_ptr<OpDesc> convert(const onnx::NodeProto &onnx_node) {
    std::shared_ptr<OpDesc> op_desc =
        std::make_shared<OpDesc>(kOpTypeTranspose);
    OnnxOpConvert::convert(onnx_node, op_desc);
    TransposeParam *param = (TransposeParam *)(op_desc->op_param_.get());
    param->perm_ = OnnxInterpret::getAttributeIntVector(onnx_node, "perm");
    return op_desc;
  };
};

REGISTER_ONNX_OP_CONVERT_IMPLEMENTION("Transpose", OnnxTransposeConvert);

}  // namespace ir
}  // namespace nndeploy