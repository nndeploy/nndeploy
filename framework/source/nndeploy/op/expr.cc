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

std::shared_ptr<Expr> makeInput(ModelDesc *model_desc, std::string name,
                                base::DataType data_type,
                                base::IntVector shape) {
  auto value_desc = std::make_shared<ValueDesc>(name, data_type, shape);
  if (model_desc != nullptr) {
    model_desc->inputs_.push_back(value_desc);
  }
  auto expr = std::make_shared<Expr>(value_desc);
  return expr;
}
void makeOutput(ModelDesc *model_desc, std::shared_ptr<Expr> expr) {
  if (model_desc != nullptr) {
    std::vector<std::string> output = expr->getOutputName();
    for (auto &name : output) {
      auto value_desc = std::make_shared<ValueDesc>(name);
      model_desc->outputs_.push_back(value_desc);
    }
  }
}
std::shared_ptr<Expr> makeBlock(ModelDesc *model_desc,
                                std::shared_ptr<ModelDesc> model_block) {
  if (model_desc != nullptr) {
    model_desc->blocks_.push_back(model_block);
  }
  auto expr = std::make_shared<Expr>(model_block);
  return expr;
}
std::shared_ptr<Expr> makeConv(ModelDesc *model_desc,
                               std::shared_ptr<Expr> input,
                               std::shared_ptr<ConvParam> param,
                               const std::string &weight,
                               const std::string &bias, std::string op_name,
                               std::string output_name) {
  std::string name = op_name;
  if (name.empty()) {
    if (model_desc != nullptr) {
      int index = model_desc->op_descs_.size();
      name = "conv" + std::to_string(index);
    } else {
      name = "conv";
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
      std::make_shared<OpDesc>(name, kOpTypeConv, inputs, outputs, param);
  if (model_desc != nullptr) {
    model_desc->op_descs_.push_back(op_desc);
  }

  auto expr = std::make_shared<Expr>(op_desc);
  return expr;
}

std::shared_ptr<Expr> makeSoftMax(ModelDesc *model_desc,
                                  std::shared_ptr<Expr> input,
                                  std::shared_ptr<SoftmaxParam> param,
                                  std::string op_name,
                                  std::string output_name) {
  std::string name = op_name;
  if (name.empty()) {
    if (model_desc != nullptr) {
      int index = model_desc->op_descs_.size();
      name = "softmax" + std::to_string(index);
    } else {
      name = "softmax";
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
  auto op_desc =
      std::make_shared<OpDesc>(name, kOpTypeSoftmax, inputs, outputs, param);
  if (model_desc != nullptr) {
    model_desc->op_descs_.push_back(op_desc);
  }

  auto expr = std::make_shared<Expr>(op_desc);
  return expr;
}

std::shared_ptr<Expr> makeAdd(ModelDesc *model_desc,
                              std::shared_ptr<Expr> input_0,
                              std::shared_ptr<Expr> input_1,
                              std::string op_name, std::string output_name) {
  std::string name = op_name;
  if (name.empty()) {
    if (model_desc != nullptr) {
      int index = model_desc->op_descs_.size();
      name = "add" + std::to_string(index);
    } else {
      name = "add";
    }
  }
  std::vector<std::string> inputs;
  inputs.emplace_back(input_0->getOutputName()[0]);
  inputs.emplace_back(input_1->getOutputName()[0]);
  // 节点输出
  std::vector<std::string> outputs;
  if (!output_name.empty()) {
    outputs.push_back(output_name);
  } else {
    outputs.push_back(name + ".output");
  }
  auto op_desc = std::make_shared<OpDesc>(name, kOpTypeAdd, inputs, outputs);
  if (model_desc != nullptr) {
    model_desc->op_descs_.push_back(op_desc);
  }

  auto expr = std::make_shared<Expr>(op_desc);
  return expr;
}

// TODO: @Leonisux:
// 补充llama的算子的手动构图函数

// void testExpr() {
//   {
//     auto model_desc = std::make_shared<ModelDesc>();
//     auto input = makeInput("input", base::dataTypeOf<float>(), {1, 3, 224,
//     224},
//                            model_desc);
//     auto conv1 =
//         makeConv(input, "weight", "bias", std::make_shared<ConvParam>(),
//                    "conv1", "conv1.output", model_desc);
//     auto relu1 = makeRelu(conv1, "relu1", "relu1.output", model_desc);
//     auto output = makeOutput(relu1, model_desc);
//   }
//   {
//     auto model_desc = std::make_shared<ModelDesc>();
//     auto input = makeInput(model_desc);
//     auto conv1 = makeConv(model_desc, input, "weight", "bias");
//     auto relu1 = makeRelu(model_desc, conv1);
//     auto output = makeOutput(model_desc, relu1);
//   }
// }

}  // namespace op
}  // namespace nndeploy