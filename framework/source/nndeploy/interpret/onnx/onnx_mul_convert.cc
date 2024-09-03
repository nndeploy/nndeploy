
#include <google/protobuf/message.h>
#include <google/protobuf/text_format.h>

#include "google/protobuf/io/coded_stream.h"
#include "google/protobuf/io/zero_copy_stream_impl.h"
#include "nndeploy/interpret/onnx/onnx_interpret.h"
#include "nndeploy/op/ir.h"
#include "nndeploy/op/op.h"

namespace nndeploy {
namespace interpret {

class OnnxMulConvert : public OnnxOpConvert {
 public:
  OnnxMulConvert() : OnnxOpConvert() {}
  virtual ~OnnxMulConvert() {}

  virtual std::shared_ptr<op::OpDesc> convert(
      const onnx::NodeProto &onnx_node) {
    std::shared_ptr<op::OpDesc> op_desc =
        std::make_shared<op::OpDesc>(op::kOpTypeMul);
    OnnxOpConvert::convert(onnx_node, op_desc);
    return op_desc;
  };
};

REGISTER_ONNX_OP_CONVERT_IMPLEMENTION("Mul", OnnxMulConvert);

}  // namespace interpret
}  // namespace nndeploy