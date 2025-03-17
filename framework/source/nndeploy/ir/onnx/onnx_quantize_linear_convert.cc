
#include <google/protobuf/message.h>
#include <google/protobuf/text_format.h>

#include "google/protobuf/io/coded_stream.h"
#include "google/protobuf/io/zero_copy_stream_impl.h"
#include "nndeploy/ir/ir.h"
#include "nndeploy/ir/onnx/onnx_interpret.h"

namespace nndeploy {
namespace ir {

class OnnxQuantizeLinearConvert : public OnnxOpConvert {
 public:
  OnnxQuantizeLinearConvert() : OnnxOpConvert() {}
  virtual ~OnnxQuantizeLinearConvert() {}

  virtual std::shared_ptr<OpDesc> convert(const onnx::NodeProto &onnx_node) {
    std::shared_ptr<OpDesc> op_desc =
        std::make_shared<OpDesc>(kOpTypeQuantizeLinear);
    OnnxOpConvert::convert(onnx_node, op_desc);

    QuantizeLinearParam *param =
        (QuantizeLinearParam *)(op_desc->op_param_.get());
    param->axis_ = OnnxInterpret::getAttributeInt(onnx_node, "axis", 1);
    param->saturate_ = OnnxInterpret::getAttributeInt(onnx_node, "saturate", 1);

    return op_desc;
  };
};

REGISTER_ONNX_OP_CONVERT_IMPLEMENTION("QuantizeLinear",
                                      OnnxQuantizeLinearConvert);

}  // namespace ir
}  // namespace nndeploy