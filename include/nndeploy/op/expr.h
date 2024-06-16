
#ifndef _NNDEPLOY_OP_EXPR_H_
#define _NNDEPLOY_OP_EXPR_H_
#include "nndeploy/base/macro.h"
#include "nndeploy/device/tensor.h"
#include "nndeploy/op/ir.h"
namespace nndeploy {

namespace op {

// 标识Expr的类型
enum ExprType : int {
  kExprTypeData = 0x0000,

  kExprTypeOpDesc,
};

/**
 * @brief 统一的创建net时候的接口
 *
 * auto  conv1=make->MakeConv(input,weight,{3,3},{1,1});
 * auto add1 = MakeBiasAdd(conv1, bias)
 * auto relu1 = MakeRelu(add1)
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
  std::string data_;
  std::shared_ptr<OpDescAndParam> op_desc_param_;
  ExprType type_;

  // std::vector<std::shared_ptr<Expr>> producers_;
  // std::vector<std::shared_ptr<Expr>> consumers_;
};
}  // namespace op

}  // namespace nndeploy

#endif