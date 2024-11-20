
#include <google/protobuf/message.h>
#include <google/protobuf/text_format.h>

#include "google/protobuf/io/coded_stream.h"
#include "google/protobuf/io/zero_copy_stream_impl.h"
#include "nndeploy/ir/ir.h"
#include "nndeploy/ir/onnx/onnx_interpret.h"

namespace nndeploy {
namespace ir {

class OnnxResizeConvert : public OnnxOpConvert {
 public:
  OnnxResizeConvert() : OnnxOpConvert() {}
  virtual ~OnnxResizeConvert() {}

  virtual std::shared_ptr<OpDesc> convert(const onnx::NodeProto &onnx_node) {
    std::shared_ptr<OpDesc> op_desc = std::make_shared<OpDesc>(kOpTypeResize);
    // NNDEPLOY_LOGE("onnx_node.size = %lld\n", onnx_node.input_size());
    OnnxOpConvert::convert(onnx_node, op_desc);
    ResizeParam *param = (ResizeParam *)(op_desc->op_param_.get());
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

}  // namespace ir
}  // namespace nndeploy