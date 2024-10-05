
#include <google/protobuf/message.h>
#include <google/protobuf/text_format.h>

#include "google/protobuf/io/coded_stream.h"
#include "google/protobuf/io/zero_copy_stream_impl.h"
#include "nndeploy/ir/ir.h"
#include "nndeploy/ir/onnx/onnx_interpret.h"

namespace nndeploy {
namespace ir {

class OnnxConvConvert : public OnnxOpConvert {
 public:
  OnnxConvConvert() : OnnxOpConvert() {}
  virtual ~OnnxConvConvert() {}

  virtual std::shared_ptr<OpDesc> convert(const onnx::NodeProto &onnx_node) {
    std::shared_ptr<OpDesc> op_desc = std::make_shared<OpDesc>(kOpTypeConv);
    OnnxOpConvert::convert(onnx_node, op_desc);
    ConvParam *param = (ConvParam *)(op_desc->op_param_.get());
    param->auto_pad_ =
        OnnxInterpret::getAttributeString(onnx_node, "auto_pad", "NOTSET");
    param->dilations_ =
        OnnxInterpret::getAttributeIntVector(onnx_node, "dilations");
    param->group_ = OnnxInterpret::getAttributeInt(onnx_node, "group", 1);
    param->kernel_shape_ =
        OnnxInterpret::getAttributeIntVector(onnx_node, "kernel_shape");
    param->pads_ = OnnxInterpret::getAttributeIntVector(onnx_node, "pads");
    param->strides_ =
        OnnxInterpret::getAttributeIntVector(onnx_node, "strides");
    return op_desc;
  };
};

REGISTER_ONNX_OP_CONVERT_IMPLEMENTION("Conv", OnnxConvConvert);

}  // namespace ir
}  // namespace nndeploy