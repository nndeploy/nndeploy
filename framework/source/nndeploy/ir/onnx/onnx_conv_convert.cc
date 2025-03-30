
#include <google/protobuf/message.h>
#include <google/protobuf/text_format.h>

#include "google/protobuf/io/coded_stream.h"
#include "google/protobuf/io/zero_copy_stream_impl.h"
#include "nndeploy/ir/ir.h"
#include "nndeploy/ir/onnx/onnx_interpret.h"

namespace nndeploy {
namespace ir {

/**
 * @brief ONNX卷积算子转换类
 * 负责将ONNX的Conv算子转换为nndeploy内部的Conv算子
 */
class OnnxConvConvert : public OnnxOpConvert {
 public:
  OnnxConvConvert() : OnnxOpConvert() {}
  virtual ~OnnxConvConvert() {}

  /**
   * @brief 转换ONNX的Conv节点为nndeploy的Conv算子描述
   * @param onnx_node ONNX的Conv节点
   * @return 转换后的Conv算子描述
   */
  virtual std::shared_ptr<OpDesc> convert(const onnx::NodeProto &onnx_node) {
    // 创建Conv算子描述
    std::shared_ptr<OpDesc> op_desc = std::make_shared<OpDesc>(kOpTypeConv);
    // 调用基类转换公共属性
    OnnxOpConvert::convert(onnx_node, op_desc);
    // 获取Conv参数指针
    ConvParam *param = (ConvParam *)(op_desc->op_param_.get());
    
    // 设置自动padding模式,默认为"NOTSET"
    param->auto_pad_ =
        OnnxInterpret::getAttributeString(onnx_node, "auto_pad", "NOTSET");
    // 设置膨胀系数
    param->dilations_ =
        OnnxInterpret::getAttributeIntVector(onnx_node, "dilations");
    // 设置分组卷积的组数,默认为1
    param->group_ = OnnxInterpret::getAttributeInt(onnx_node, "group", 1);
    // 设置卷积核大小
    param->kernel_shape_ =
        OnnxInterpret::getAttributeIntVector(onnx_node, "kernel_shape");
    // 设置padding值
    param->pads_ = OnnxInterpret::getAttributeIntVector(onnx_node, "pads");
    // 设置步长
    param->strides_ =
        OnnxInterpret::getAttributeIntVector(onnx_node, "strides");
    
    return op_desc;
  };
};

REGISTER_ONNX_OP_CONVERT_IMPLEMENTION("Conv", OnnxConvConvert);

}  // namespace ir
}  // namespace nndeploy