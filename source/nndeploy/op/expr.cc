#include "nndeploy/op/expr.h"

namespace nndeploy {
namespace op {

std::vector<std::string> Expr::getOutputName() {
  switch (expr_type_) {
    case kExprTypeValueDesc:
      return {value_desc_.name_};
    case kExprTypeOpDescAndParam:
      return op_desc_param_->op_desc_.outputs_;
    default:
      return {""};
  }
}

std::shared_ptr<Expr> ModelDesc::MakeConv2d(
    std::shared_ptr<Expr> input,
    std::shared_ptr<Expr> weight,  // 卷积还有bias呀
    std::shared_ptr<Conv2dParam> param, std::string name) {
  auto descAndParam = std::make_shared<OpDescAndParam>();
  descAndParam->op_desc_ = OpDesc(name, kOpTypeConv2d);
  descAndParam->op_desc_.inputs_ = {input->getOutputName()[0],
                                    weight->getOutputName()[0]};
  descAndParam->op_desc_.outputs_ = {
      name +
      ".output"};  // TODO：假如想让输出保持模型结构一致，是不是需要新增一个参数呢？
  descAndParam->op_param_ = param;

  auto convExpr = std::make_shared<Expr>(descAndParam);

  op_desc_params_.push_back(descAndParam);

  return convExpr;
}

std::shared_ptr<Expr> ModelDesc::MakeRelu(std::shared_ptr<Expr> input,
                                          std::string name) {
  auto descAndParam = std::make_shared<OpDescAndParam>();
  descAndParam->op_desc_ = OpDesc(name, kOpTypeRelu);
  descAndParam->op_desc_.inputs_ = {input->getOutputName()};
  descAndParam->op_desc_.outputs_ = {name + ".output"};

  auto reluExpr = std::make_shared<Expr>(descAndParam);

  op_desc_params_.push_back(descAndParam);

  return reluExpr;
}

void testExpr() {
  {
    auto model_desc = std::make_shared<ModelDesc>();
    auto input = MakeInput("input", base::dataTypeOf<float>(), {1, 3, 224, 224},
                           model_desc);
    auto conv1 =
        MakeConv2d(input, "weight", "bias", std::make_shared<Conv2dParam>(),
                   "conv1", "conv1.output", model_desc);
    auto relu1 = MakeRelu(conv1, "relu1", "relu1.output", model_desc);
    auto output = MakeOutput(relu1, model_desc);
  }
  {
    auto model_desc = std::make_shared<ModelDesc>();
    auto input = MakeInput(model_desc);
    auto conv1 = MakeConv2d(model_desc, input, "weight", "bias");
    auto relu1 = MakeRelu(model_desc, conv1);
    auto output = MakeOutput(model_desc, relu1);
  }
}

class VAE {
 public:
  VAE() {
    auto model_desc = std::make_shared<ModelDesc>();
    auto input = MakeInput(model_desc);
    auto conv1 = MakeConv2d(model_desc, input, "weight", "bias");
    auto relu1 = MakeRelu(model_desc, conv1);
    auto output = MakeOutput(model_desc, relu1);
  }
};

}  // namespace op
}  // namespace nndeploy