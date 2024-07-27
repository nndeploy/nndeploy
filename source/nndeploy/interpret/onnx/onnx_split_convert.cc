
#include <google/protobuf/message.h>
#include <google/protobuf/text_format.h>

#include "google/protobuf/io/coded_stream.h"
#include "google/protobuf/io/zero_copy_stream_impl.h"
#include "nndeploy/interpret/onnx/onnx_interpret.h"
#include "nndeploy/op/ir.h"
#include "nndeploy/op/op.h"

namespace nndeploy {
namespace interpret {

class OnnxSplitConvert : public OnnxOpConvert {
 public:
  OnnxSplitConvert() : OnnxOpConvert() {}
  virtual ~OnnxSplitConvert() {}

  virtual std::shared_ptr<op::OpDesc> convert(
      const onnx::NodeProto &onnx_node) {
    std::shared_ptr<op::OpDesc> op_desc =
        std::make_shared<op::OpDesc>(op::kOpTypeSplit);
    OnnxOpConvert::convert(onnx_node, op_desc);
    op::SplitParam *param = (op::SplitParam *)(op_desc->op_param_.get());
    param->axis_ = OnnxInterpret::getAttributeInt(onnx_node, "axis", 0);
    param->num_outputs_ =
        OnnxInterpret::getAttributeInt(onnx_node, "num_outputs", INT_MAX);
    return op_desc;
  };
};

REGISTER_ONNX_OP_CONVERT_IMPLEMENTION("Split", OnnxSplitConvert);

}  // namespace interpret
}  // namespace nndeploy