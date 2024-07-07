
#ifndef _NNDEPLOY_OP_EXPR_H_
#define _NNDEPLOY_OP_EXPR_H_

#include "nndeploy/base/macro.h"
#include "nndeploy/device/tensor.h"
#include "nndeploy/op/base/conv2d.h"
#include "nndeploy/op/base/relu.h"
#include "nndeploy/op/ir.h"

namespace nndeploy {
namespace op {

// 标识Expr的类型
enum ExprType : int {
  // TODO：需要细分为输入、输出、权重吗？似乎现在这样更加简单且直观，但是只有这个信息能区分输入与权重吗？（应该可以？）
  kExprTypeValueDesc = 0x0000,

  kExprTypeOpDescAndParam,
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
  Expr(const std::string &name) : expr_type_(kExprTypeValueDesc) {}
  Expr(const std::string &name, base::DataType data_type)
      : expr_type_(kExprTypeValueDesc) {}
  Expr(const std::string &name, base::DataType data_type, base::IntVector shape)
      : expr_type_(kExprTypeValueDesc) {}
  Expr(const ValueDesc &value_desc) : expr_type_(kExprTypeValueDesc) {}

  Expr(std::shared_ptr<OpDescAndParam> op_desc_param)
      : op_desc_param_(op_desc_param), expr_type_(kExprTypeOpDescAndParam) {}

  std::vector<std::string> getOutputName();

 protected:
  ExprType expr_type_;
  ValueDesc value_desc_;
  std::shared_ptr<OpDescAndParam> op_desc_param_;
};

/**
 * @brief  一系列创建函数
 */
// input
std::shared_ptr<Expr> MakeInput(std::string name, base::DataType data_type,
                                base::IntVector shape,
                                std::shared_ptr<op::ModelDesc> model_desc);
// output
std::shared_ptr<Expr> MakeOutput(std::shared_ptr<Expr> expr,
                                 std::shared_ptr<op::ModelDesc> model_desc);
// conv2d
std::shared_ptr<Expr> MakeConv2d(std::shared_ptr<Expr> input,
                                 const std::string &weight,
                                 const std::string &bias,
                                 std::shared_ptr<Conv2dParam> param,
                                 std::string op_name, std::string output_name,
                                 std::shared_ptr<op::ModelDesc> model_desc);
// relu
std::shared_ptr<Expr> MakeRelu(std::shared_ptr<Expr> input, std::string op_name,
                               std::string output_name,
                               std::shared_ptr<op::ModelDesc> model_desc);

}  // namespace op
}  // namespace nndeploy

#endif