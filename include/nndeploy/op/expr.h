
#ifndef _NNDEPLOY_OP_EXPR_H_
#define _NNDEPLOY_OP_EXPR_H_

#include "nndeploy/base/macro.h"
#include "nndeploy/device/tensor.h"
#include "nndeploy/op/ir.h"

namespace nndeploy {
namespace op {

// 标识Expr的类型
enum ExprType : int {
  // TODO：需要细分为输入、输出、权重吗？似乎现在这样更加简单且直观，但是只有这个信息能区分输入与权重吗？（应该可以？）
  kExprTypeData = 0x0000,

  kExprTypeOpDesc,
};

/**
 * @brief 统一的创建net时候的接口
 * std::string input_name = "images"; TODO:
 * auto input = MakeInput(input_name); // TODO:需要增加这个接口吗
 * auto conv1=make->MakeConv(input,weight,{3,3},{1,1});
 * auto add1 = MakeBiasAdd(conv1, bias)
 * auto relu1 = MakeRelu(add1)
 * auto output = MakeOutput(relu1); // TODO:需要增加这个接口吗
 *
 * 将数据和op统一为一个容器，然后在net初始化时利用Expr的信息构图
 */
class NNDEPLOY_CC_API Expr {
 public:
  Expr(std::string data) : data_(data), type_(kExprTypeData) {}

  Expr(std::shared_ptr<OpDescAndParam> op_desc)
      : op_desc_param_(op_desc), type_(kExprTypeOpDesc) {}

  std::vector<std::string> getOutputName();

 protected:
  std::string data_;  // TODO：这个需要用ValueDesc来表示吗？
  std::shared_ptr<OpDescAndParam> op_desc_param_;
  ExprType type_;

  // std::vector<std::shared_ptr<Expr>> producers_;
  // std::vector<std::shared_ptr<Expr>> consumers_;
};

}  // namespace op
}  // namespace nndeploy

#endif