
#include <google/protobuf/message.h>
#include <google/protobuf/text_format.h>

#include "google/protobuf/io/coded_stream.h"
#include "google/protobuf/io/zero_copy_stream_impl.h"
#include "nndeploy/interpret/onnx/onnx_interpret.h"
#include "nndeploy/op/ir.h"
#include "nndeploy/op/op.h"

namespace nndeploy {
namespace interpret {

class OnnxSubConvert : public OnnxOpConvert {
 public:
  OnnxSubConvert() : OnnxOpConvert() {}
  virtual ~OnnxSubConvert() {}

  virtual std::shared_ptr<op::OpDesc> convert(
      const onnx::NodeProto &onnx_node) {
    std::shared_ptr<op::OpDesc> op_desc =
        std::make_shared<op::OpDesc>(op::kOpTypeSub);
    OnnxOpConvert::convert(onnx_node, op_desc);
    return op_desc;
  };
};

REGISTER_ONNX_OP_CONVERT_IMPLEMENTION("Sub", OnnxSubConvert);

}  // namespace interpret
}  // namespace nndeploy