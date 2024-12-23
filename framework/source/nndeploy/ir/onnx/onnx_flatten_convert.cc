
#include <google/protobuf/message.h>
#include <google/protobuf/text_format.h>

#include "google/protobuf/io/coded_stream.h"
#include "google/protobuf/io/zero_copy_stream_impl.h"
#include "nndeploy/ir/ir.h"
#include "nndeploy/ir/onnx/onnx_interpret.h"

namespace nndeploy {
namespace ir {

class OnnxOnnxFlattenConvert : public OnnxOpConvert {
 public:
  OnnxOnnxFlattenConvert() : OnnxOpConvert() {}
  virtual ~OnnxOnnxFlattenConvert() {}

  virtual std::shared_ptr<OpDesc> convert(const onnx::NodeProto &onnx_node) {
    std::shared_ptr<OpDesc> op_desc = std::make_shared<OpDesc>(kOpTypeFlatten);
    OnnxOpConvert::convert(onnx_node, op_desc);
    FlattenParam *param = (FlattenParam *)(op_desc->op_param_.get());
    param->axis_ =
        OnnxInterpret::getAttributeInt(onnx_node, "axis", (int32_t)1);
    return op_desc;
  };
};

REGISTER_ONNX_OP_CONVERT_IMPLEMENTION("Flatten", OnnxOnnxFlattenConvert);

}  // namespace ir
}  // namespace nndeploy