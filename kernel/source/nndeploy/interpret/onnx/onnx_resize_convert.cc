
#include <google/protobuf/message.h>
#include <google/protobuf/text_format.h>

#include "google/protobuf/io/coded_stream.h"
#include "google/protobuf/io/zero_copy_stream_impl.h"
#include "nndeploy/interpret/onnx/onnx_interpret.h"
#include "nndeploy/op/ir.h"
#include "nndeploy/op/op.h"

namespace nndeploy {
namespace interpret {

class OnnxResizeConvert : public OnnxOpConvert {
 public:
  OnnxResizeConvert() : OnnxOpConvert() {}
  virtual ~OnnxResizeConvert() {}

  virtual std::shared_ptr<op::OpDesc> convert(
      const onnx::NodeProto &onnx_node) {
    std::shared_ptr<op::OpDesc> op_desc =
        std::make_shared<op::OpDesc>(op::kOpTypeResize);
    OnnxOpConvert::convert(onnx_node, op_desc);
    op::ResizeParam *param = (op::ResizeParam *)(op_desc->op_param_.get());
    param->antialias_ =
        OnnxInterpret::getAttributeInt(onnx_node, "antialias", 0);
    param->axes_ = OnnxInterpret::getAttributeInt(onnx_node, "axes", INT_MAX);
    param->coordinate_transformation_mode_ = OnnxInterpret::getAttributeString(
        onnx_node, "coordinate_transformation_mode", "half_pixel");
    param->cubic_coeff_a_ =
        OnnxInterpret::getAttributeFloat(onnx_node, "cubic_coeff_a", -0.75);
    param->exclude_outside_ =
        OnnxInterpret::getAttributeInt(onnx_node, "exclude_outside", 0);
    param->extrapolation_value_ = OnnxInterpret::getAttributeFloat(
        onnx_node, "extrapolation_value", -0.0);
    param->keep_aspect_ratio_policy_ = OnnxInterpret::getAttributeString(
        onnx_node, "keep_aspect_ratio_policy", "stretch");
    param->mode_ =
        OnnxInterpret::getAttributeString(onnx_node, "mode", "nearest");
    param->nearest_mode_ = OnnxInterpret::getAttributeString(
        onnx_node, "nearest_mode", "round_prefer_floor");
    return op_desc;
  };
};

REGISTER_ONNX_OP_CONVERT_IMPLEMENTION("Resize", OnnxResizeConvert);

}  // namespace interpret
}  // namespace nndeploy