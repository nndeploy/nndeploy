
#include <google/protobuf/message.h>
#include <google/protobuf/text_format.h>

#include "google/protobuf/io/coded_stream.h"
#include "google/protobuf/io/zero_copy_stream_impl.h"
#include "nndeploy/interpret/onnx/onnx_interpret.h"
#include "nndeploy/op/ir.h"
#include "nndeploy/op/op.h"

namespace nndeploy {
namespace interpret {

class OnnxMaxPoolConvert : public OnnxOpConvert {
 public:
  OnnxMaxPoolConvert() : OnnxOpConvert() {}
  virtual ~OnnxMaxPoolConvert() {}

  virtual std::shared_ptr<op::OpDesc> convert(
      const onnx::NodeProto &onnx_node) {
    std::shared_ptr<op::OpDesc> op_desc =
        std::make_shared<op::OpDesc>(op::kOpTypeMaxPool);
    OnnxOpConvert::convert(onnx_node, op_desc);
    op::MaxPoolParam *param = (op::MaxPoolParam *)(op_desc->op_param_.get());
    param->auto_pad_ =
        OnnxInterpret::getAttributeString(onnx_node, "auto_pad", "NOTSET");
    param->ceil_mode_ =
        OnnxInterpret::getAttributeInt(onnx_node, "ceil_mode", 0);
    OnnxInterpret::getAttributeIntVector(onnx_node, "dilations");
    param->kernel_shape_ =
        OnnxInterpret::getAttributeIntVector(onnx_node, "kernel_shape");
    param->pads_ = OnnxInterpret::getAttributeIntVector(onnx_node, "pads");
    param->storage_order_ =
        OnnxInterpret::getAttributeInt(onnx_node, "storage_order", 0);
    param->strides_ =
        OnnxInterpret::getAttributeIntVector(onnx_node, "strides");
    return op_desc;
  };
};

REGISTER_ONNX_OP_CONVERT_IMPLEMENTION("MaxPool", OnnxMaxPoolConvert);

}  // namespace interpret
}  // namespace nndeploy