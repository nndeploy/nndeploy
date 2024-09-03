
#include <google/protobuf/message.h>
#include <google/protobuf/text_format.h>

#include "google/protobuf/io/coded_stream.h"
#include "google/protobuf/io/zero_copy_stream_impl.h"
#include "nndeploy/interpret/onnx/onnx_interpret.h"
#include "nndeploy/op/ir.h"
#include "nndeploy/op/op.h"

namespace nndeploy {
namespace interpret {

class OnnxTransposeConvert : public OnnxOpConvert {
 public:
  OnnxTransposeConvert() : OnnxOpConvert() {}
  virtual ~OnnxTransposeConvert() {}

  virtual std::shared_ptr<op::OpDesc> convert(
      const onnx::NodeProto &onnx_node) {
    std::shared_ptr<op::OpDesc> op_desc =
        std::make_shared<op::OpDesc>(op::kOpTypeTranspose);
    OnnxOpConvert::convert(onnx_node, op_desc);
    op::TransposeParam *param =
        (op::TransposeParam *)(op_desc->op_param_.get());
    param->perm_ = OnnxInterpret::getAttributeIntVector(onnx_node, "perm");
    return op_desc;
  };
};

REGISTER_ONNX_OP_CONVERT_IMPLEMENTION("Transpose", OnnxTransposeConvert);

}  // namespace interpret
}  // namespace nndeploy