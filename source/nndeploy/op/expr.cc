#include "nndeploy/op/expr.h"

namespace nndeploy {
namespace op {

Expr::Expr(const std::string &name) : expr_type_(kExprTypeValueDesc) {
  value_desc_ = std::make_shared<ValueDesc>(name);
}
Expr::Expr(const std::string &name, base::DataType data_type)
    : expr_type_(kExprTypeValueDesc) {
  value_desc_ = std::make_shared<ValueDesc>(name, data_type);
}
Expr::Expr(const std::string &name, base::DataType data_type,
           base::IntVector shape)
    : expr_type_(kExprTypeValueDesc) {
  value_desc_ = std::make_shared<ValueDesc>(name, data_type, shape);
}
Expr::Expr(std::shared_ptr<ValueDesc> value_desc)
    : expr_type_(kExprTypeValueDesc), value_desc_(value_desc) {}
Expr::Expr(std::shared_ptr<OpDesc> op_desc)
    : expr_type_(kExprTypeOpDesc), op_desc_(op_desc) {}
Expr::Expr(std::shared_ptr<ModelDesc> model_desc)
    : expr_type_(kExprTypeOpDesc), model_desc_(model_desc) {}

std::vector<std::string> Expr::getOutputName() {
  switch (expr_type_) {
    case kExprTypeValueDesc:
      return {value_desc_->name_};
    case kExprTypeOpDesc:
      return op_desc_->outputs_;
    case kExprTypeModelDesc: {
      std::vector<std::string> outputs;
      for (auto &output : model_desc_->outputs_) {
        outputs.push_back(output->name_);
      }
      return outputs;
    }
    default:
      return {""};
  }
}

std::shared_ptr<Expr> MakeInput(std::shared_ptr<op::ModelDesc> model_desc,
                                std::string name, base::DataType data_type,
                                base::IntVector shape) {
  auto value_desc = std::make_shared<ValueDesc>(name, data_type, shape);
  if (model_desc != nullptr) {
    model_desc->inputs_.push_back(value_desc);
  }
  auto expr = std::make_shared<Expr>(value_desc);
  return expr;
}
void MakeOutput(std::shared_ptr<op::ModelDesc> model_desc,
                std::shared_ptr<Expr> expr) {
  if (model_desc != nullptr) {
    std::vector<std::string> output = expr->getOutputName();
    for (auto &name : output) {
      auto value_desc = std::make_shared<ValueDesc>(name);
      model_desc->outputs_.push_back(value_desc);
    }
  }
}
std::shared_ptr<Expr> MakeConv2d(std::shared_ptr<op::ModelDesc> model_desc,
                                 std::shared_ptr<Expr> input,
                                 std::shared_ptr<Conv2dParam> param,
                                 const std::string &weight,
                                 const std::string &bias, std::string op_name,
                                 std::string output_name) {
  std::string name = op_name;
  if (name.empty()) {
    if (model_desc != nullptr) {
      int index = model_desc->op_descs_.size();
      name = "conv2d" + std::to_string(index);
    } else {
      name = "conv2d";
    }
  }
  std::vector<std::string> inputs = {input->getOutputName()[0]};
  if (!weight.empty()) {
    inputs.push_back(weight);
  }
  if (!bias.empty()) {
    inputs.push_back(bias);
  }
  // 节点输出
  std::vector<std::string> outputs;
  if (!output_name.empty()) {
    outputs.push_back(output_name);
  } else {
    outputs.push_back(name + ".output");
  }
  auto op_desc =
      std::make_shared<OpDesc>(name, kOpTypeConv2d, inputs, outputs, param);
  if (model_desc != nullptr) {
    model_desc->op_descs_.push_back(op_desc);
  }

  auto expr = std::make_shared<Expr>(op_desc);
  return expr;
}

std::shared_ptr<Expr> MakeRelu(std::shared_ptr<op::ModelDesc> model_desc,
                               std::shared_ptr<Expr> input, std::string op_name,
                               std::string output_name) {
  std::string name = op_name;
  if (name.empty()) {
    if (model_desc != nullptr) {
      int index = model_desc->op_descs_.size();
      name = "relu" + std::to_string(index);
    } else {
      name = "relu";
    }
  }
  std::vector<std::string> inputs = {input->getOutputName()[0]};
  // 节点输出
  std::vector<std::string> outputs;
  if (!output_name.empty()) {
    outputs.push_back(output_name);
  } else {
    outputs.push_back(name + ".output");
  }
  auto op_desc = std::make_shared<OpDesc>(name, kOpTypeRelu, inputs, outputs);
  if (model_desc != nullptr) {
    model_desc->op_descs_.push_back(op_desc);
  }

  auto expr = std::make_shared<Expr>(op_desc);
  return expr;
}

// void testExpr() {
//   {
//     auto model_desc = std::make_shared<ModelDesc>();
//     auto input = MakeInput("input", base::dataTypeOf<float>(), {1, 3, 224,
//     224},
//                            model_desc);
//     auto conv1 =
//         MakeConv2d(input, "weight", "bias", std::make_shared<Conv2dParam>(),
//                    "conv1", "conv1.output", model_desc);
//     auto relu1 = MakeRelu(conv1, "relu1", "relu1.output", model_desc);
//     auto output = MakeOutput(relu1, model_desc);
//   }
//   {
//     auto model_desc = std::make_shared<ModelDesc>();
//     auto input = MakeInput(model_desc);
//     auto conv1 = MakeConv2d(model_desc, input, "weight", "bias");
//     auto relu1 = MakeRelu(model_desc, conv1);
//     auto output = MakeOutput(model_desc, relu1);
//   }
// }

class VAE {
 public:
  VAE() {
    auto model_desc = std::make_shared<ModelDesc>();
    auto input = MakeInput(model_desc, "input");
    auto conv1 = MakeConv2d(model_desc, input, std::make_shared<Conv2dParam>(),
                            "weight", "bias");
    auto relu1 = MakeRelu(model_desc, conv1);
    MakeOutput(model_desc, relu1);
  }
};

}  // namespace op
}  // namespace nndeploy