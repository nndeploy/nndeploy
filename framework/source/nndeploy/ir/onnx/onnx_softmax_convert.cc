
#include <google/protobuf/message.h>
#include <google/protobuf/text_format.h>

#include "google/protobuf/io/coded_stream.h"
#include "google/protobuf/io/zero_copy_stream_impl.h"
#include "nndeploy/ir/ir.h"
#include "nndeploy/ir/onnx/onnx_interpret.h"

namespace nndeploy {
namespace ir {

class OnnxSoftmaxConvert : public OnnxOpConvert {
 public:
  OnnxSoftmaxConvert() : OnnxOpConvert() {}
  virtual ~OnnxSoftmaxConvert() {}

  virtual std::shared_ptr<OpDesc> convert(const onnx::NodeProto &onnx_node) {
    std::shared_ptr<OpDesc> op_desc = std::make_shared<OpDesc>(kOpTypeSoftmax);
    OnnxOpConvert::convert(onnx_node, op_desc);
    SoftmaxParam *param = (SoftmaxParam *)(op_desc->op_param_.get());
    param->axis_ = OnnxInterpret::getAttributeInt(onnx_node, "axis", -1);
    return op_desc;
  };
};

REGISTER_ONNX_OP_CONVERT_IMPLEMENTION("Softmax", OnnxSoftmaxConvert);

}  // namespace ir
}  // namespace nndeploy