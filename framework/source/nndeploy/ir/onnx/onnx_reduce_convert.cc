/**
 * Reduce类算子的转换，这类算子的属性和输入处理方式相似，包含：
 * ReduceMax
 * ReduceMin
 * ReduceSum
 * ReduceMean
 */

#include <google/protobuf/message.h>
#include <google/protobuf/text_format.h>

#include "google/protobuf/io/coded_stream.h"
#include "google/protobuf/io/zero_copy_stream_impl.h"
#include "nndeploy/ir/ir.h"
#include "nndeploy/ir/onnx/onnx_interpret.h"

namespace nndeploy {
namespace ir {
namespace {
#define REGISTER_ONNX_OP_CONVERT_CLASS(OP_NAME, OP_TYPE, OP_PARAM)         \
  class Onnx##OP_NAME##Convert : public OnnxOpConvert {                    \
   public:                                                                 \
    Onnx##OP_NAME##Convert() : OnnxOpConvert() {}                          \
    virtual ~Onnx##OP_NAME##Convert() {}                                   \
                                                                           \
    virtual std::shared_ptr<OpDesc> convert(                               \
        const onnx::NodeProto &onnx_node) {                                \
      std::shared_ptr<OpDesc> op_desc = std::make_shared<OpDesc>(OP_TYPE); \
      OnnxOpConvert::convert(onnx_node, op_desc);                          \
      OP_PARAM *param = (OP_PARAM *)(op_desc->op_param_.get());            \
                                                                           \
      param->keepdims_ =                                                   \
          OnnxInterpret::getAttributeInt(onnx_node, "keepdims", 1);        \
      param->noop_with_empty_axes_ = OnnxInterpret::getAttributeInt(       \
          onnx_node, "noop_with_empty_axes", 0);                           \
                                                                           \
      return op_desc;                                                      \
    }                                                                      \
  };                                                                       \
                                                                           \
  REGISTER_ONNX_OP_CONVERT_IMPLEMENTION(#OP_NAME, Onnx##OP_NAME##Convert)
}  // namespace

REGISTER_ONNX_OP_CONVERT_CLASS(ReduceMean, kOpTypeReduceMean, ReduceMeanParam);
REGISTER_ONNX_OP_CONVERT_CLASS(ReduceMax, kOpTypeReduceMax, ReduceMaxParam);
REGISTER_ONNX_OP_CONVERT_CLASS(ReduceMin, kOpTypeReduceMin, ReduceMinParam);
REGISTER_ONNX_OP_CONVERT_CLASS(ReduceSum, kOpTypeReduceSum, ReduceSumParam);

}  // namespace ir
}  // namespace nndeploy