
#include <google/protobuf/message.h>
#include <google/protobuf/text_format.h>

#include "google/protobuf/io/coded_stream.h"
#include "google/protobuf/io/zero_copy_stream_impl.h"
#include "nndeploy/ir/ir.h"
#include "nndeploy/ir/onnx/onnx_interpret.h"

namespace nndeploy {
namespace ir {

class OnnxOnnxBatchNormalizationConvert : public OnnxOpConvert {
 public:
  OnnxOnnxBatchNormalizationConvert() : OnnxOpConvert() {}
  virtual ~OnnxOnnxBatchNormalizationConvert() {}

  virtual std::shared_ptr<OpDesc> convert(const onnx::NodeProto &onnx_node) {
    std::shared_ptr<OpDesc> op_desc = std::make_shared<OpDesc>(kOpTypeBatchNormalization);
    OnnxOpConvert::convert(onnx_node, op_desc);
    BatchNormalizationParam *param = (BatchNormalizationParam *)(op_desc->op_param_.get());
    param->epsilon_ =
        OnnxInterpret::getAttributeFloat(onnx_node, "epsilon", 1e-05f);
    param->momentum_ =
        OnnxInterpret::getAttributeFloat(onnx_node, "momentum", 0.9f);
    param->training_mode_ =
        OnnxInterpret::getAttributeInt(onnx_node, "training_mode", 0);
    return op_desc;
  };
};

REGISTER_ONNX_OP_CONVERT_IMPLEMENTION("BatchNormalization", OnnxOnnxBatchNormalizationConvert);

}  // namespace ir
}  // namespace nndeploy