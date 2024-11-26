
#include <google/protobuf/message.h>
#include <google/protobuf/text_format.h>

#include "google/protobuf/io/coded_stream.h"
#include "google/protobuf/io/zero_copy_stream_impl.h"
#include "nndeploy/ir/ir.h"
#include "nndeploy/ir/onnx/onnx_interpret.h"

namespace nndeploy {
namespace ir {

class OnnxOnnxGemmConvert : public OnnxOpConvert {
 public:
  OnnxOnnxGemmConvert() : OnnxOpConvert() {}
  virtual ~OnnxOnnxGemmConvert() {}

  virtual std::shared_ptr<OpDesc> convert(const onnx::NodeProto &onnx_node) {
    std::shared_ptr<OpDesc> op_desc = std::make_shared<OpDesc>(kOpTypeGemm);
    OnnxOpConvert::convert(onnx_node, op_desc);
    GemmParam *param = (GemmParam *)(op_desc->op_param_.get());
    param->alpha_ =
        OnnxInterpret::getAttributeFloat(onnx_node, "alpha", 1.0f);
    param->beta_ =
        OnnxInterpret::getAttributeFloat(onnx_node, "beta", 1.0f);
    param->trans_a_ =
        OnnxInterpret::getAttributeInt(onnx_node, "transA", (int32_t)0);
    param->trans_b_ =
        OnnxInterpret::getAttributeInt(onnx_node, "transB", (int32_t)0);
    return op_desc;
  };
};

REGISTER_ONNX_OP_CONVERT_IMPLEMENTION("Gemm", OnnxOnnxGemmConvert);

}  // namespace ir
}  // namespace nndeploy