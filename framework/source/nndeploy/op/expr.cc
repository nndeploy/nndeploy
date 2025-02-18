#include "nndeploy/ir/op_param.h"
#include "nndeploy/op/expr.h"

namespace nndeploy {
namespace op {

Expr::~Expr() {};

Expr::Expr(const std::string &name) : expr_type_(kExprTypeValueDesc) {
  value_desc_ = std::make_shared<ir::ValueDesc>(name);
}
Expr::Expr(const std::string &name, base::DataType data_type)
    : expr_type_(kExprTypeValueDesc) {
  value_desc_ = std::make_shared<ir::ValueDesc>(name, data_type);
}
Expr::Expr(const std::string &name, base::DataType data_type,
           base::IntVector shape)
    : expr_type_(kExprTypeValueDesc) {
  value_desc_ = std::make_shared<ir::ValueDesc>(name, data_type, shape);
}
Expr::Expr(std::shared_ptr<ir::ValueDesc> value_desc)
    : expr_type_(kExprTypeValueDesc), value_desc_(value_desc) {}
Expr::Expr(std::shared_ptr<ir::OpDesc> op_desc)
    : expr_type_(kExprTypeOpDesc), op_desc_(op_desc) {}
Expr::Expr(std::shared_ptr<ir::ModelDesc> model_desc)
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

std::shared_ptr<Expr> makeInput(ir::ModelDesc *model_desc, std::string name,
                                base::DataType data_type,
                                base::IntVector shape) {
  auto value_desc = std::make_shared<ir::ValueDesc>(name, data_type, shape);
  if (model_desc != nullptr) {
    model_desc->inputs_.push_back(value_desc);
  }
  auto expr = std::make_shared<Expr>(value_desc);
  return expr;
}
void makeOutput(ir::ModelDesc *model_desc, std::shared_ptr<Expr> expr) {
  if (model_desc != nullptr) {
    std::vector<std::string> output = expr->getOutputName();
    for (auto &name : output) {
      auto value_desc = std::make_shared<ir::ValueDesc>(name);
      model_desc->outputs_.push_back(value_desc);
    }
  }
}
std::shared_ptr<Expr> makeBlock(ir::ModelDesc *model_desc,
                                std::shared_ptr<ir::ModelDesc> model_block) {
  if (model_desc != nullptr && model_block != nullptr) {
    // 添加元数据
    for (auto &metadata : model_block->metadata_) {
      model_desc->metadata_[metadata.first] = metadata.second;
    }
    // 添加权重 - 采用拷贝方式
    for (auto &weight : model_block->weights_) {
      device::Tensor *tensor = weight.second->clone();
      model_desc->weights_[weight.first] = tensor;
    }
    // 添加算子
    for (auto &op : model_block->op_descs_) {
      model_desc->op_descs_.push_back(op);
    }
    // 添加中间值
    for (auto &value : model_block->values_) {
      model_desc->values_.push_back(value);
    }
  }
  auto expr = std::make_shared<Expr>(model_block);
  return expr;
}
std::shared_ptr<Expr> makeConv(ir::ModelDesc *model_desc,
                               std::shared_ptr<Expr> input,
                               std::shared_ptr<ir::ConvParam> param,
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
  auto op_desc = std::make_shared<ir::OpDesc>(name, ir::kOpTypeConv, inputs,
                                              outputs, param);
  if (model_desc != nullptr) {
    model_desc->op_descs_.push_back(op_desc);
  }

  auto expr = std::make_shared<Expr>(op_desc);
  return expr;
}

std::shared_ptr<Expr> makeSoftMax(ir::ModelDesc *model_desc,
                                  std::shared_ptr<Expr> input,
                                  std::shared_ptr<ir::SoftmaxParam> param,
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
  auto op_desc = std::make_shared<ir::OpDesc>(name, ir::kOpTypeSoftmax, inputs,
                                              outputs, param);
  if (model_desc != nullptr) {
    model_desc->op_descs_.push_back(op_desc);
  }

  auto expr = std::make_shared<Expr>(op_desc);
  return expr;
}

std::shared_ptr<Expr> makeAdd(ir::ModelDesc *model_desc,
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
  auto op_desc =
      std::make_shared<ir::OpDesc>(name, ir::kOpTypeAdd, inputs, outputs);
  if (model_desc != nullptr) {
    model_desc->op_descs_.push_back(op_desc);
  }

  auto expr = std::make_shared<Expr>(op_desc);
  return expr;
}

std::shared_ptr<Expr> makeMul(ir::ModelDesc *model_desc,
                              std::shared_ptr<Expr> input_0,
                              std::shared_ptr<Expr> input_1,
                              std::string op_name, std::string output_name) {
  std::string name = op_name;
  if (name.empty()) {
    if (model_desc != nullptr) {
      int index = model_desc->op_descs_.size();
      name = "mul" + std::to_string(index);
    } else {
      name = "mul";
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
  auto op_desc =
      std::make_shared<ir::OpDesc>(name, ir::kOpTypeMul, inputs, outputs);
  if (model_desc != nullptr) {
    model_desc->op_descs_.push_back(op_desc);
  }

  auto expr = std::make_shared<Expr>(op_desc);
  return expr;
}

std::shared_ptr<Expr> makeRelu(ir::ModelDesc *model_desc,
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
  auto op_desc =
      std::make_shared<ir::OpDesc>(name, ir::kOpTypeRelu, inputs, outputs);
  if (model_desc != nullptr) {
    model_desc->op_descs_.push_back(op_desc);
  }

  auto expr = std::make_shared<Expr>(op_desc);
  return expr;
}

// sigmoid
std::shared_ptr<Expr> makeSigmoid(ir::ModelDesc *model_desc,
                                  std::shared_ptr<Expr> input,
                                  std::string op_name,
                                  std::string output_name) {
  std::string name = op_name;
  if (name.empty()) {
    if (model_desc != nullptr) {
      int index = model_desc->op_descs_.size();
      name = "sigmoid" + std::to_string(index);
    } else {
      name = "sigmoid";
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
      std::make_shared<ir::OpDesc>(name, ir::kOpTypeSigmoid, inputs, outputs);
  if (model_desc != nullptr) {
    model_desc->op_descs_.push_back(op_desc);
  }

  auto expr = std::make_shared<Expr>(op_desc);
  return expr;
}

// silu
std::shared_ptr<Expr> makeSilu(ir::ModelDesc *model_desc,
                               std::shared_ptr<Expr> input, std::string op_name,
                               std::string output_name) {
  auto sigmoid = makeSigmoid(model_desc, input);
  auto mul = makeMul(model_desc, input, sigmoid);
  return mul;
}

// batchnorm
std::shared_ptr<Expr> makeBatchNorm(
    ir::ModelDesc *model_desc, std::shared_ptr<Expr> input,
    std::shared_ptr<ir::BatchNormalizationParam> param,
    const std::string &scale, const std::string &bias, const std::string &mean,
    const std::string &var, std::string op_name, std::string output_name) {
  std::string name = op_name;
  if (name.empty()) {
    if (model_desc != nullptr) {
      int index = model_desc->op_descs_.size();
      name = "batchnorm" + std::to_string(index);
    } else {
      name = "batchnorm";
    }
  }
  std::vector<std::string> inputs = {input->getOutputName()[0]};
  if (!scale.empty()) {
    inputs.push_back(scale);
  }
  if (!bias.empty()) {
    inputs.push_back(bias);
  }
  if (!mean.empty()) {
    inputs.push_back(mean);
  }
  if (!bias.empty()) {
    inputs.push_back(var);
  }

  std::vector<std::string> outputs;
  if (!output_name.empty()) {
    outputs.push_back(output_name);
  } else {
    outputs.push_back(name + ".output");
  }
  auto op_desc = std::make_shared<ir::OpDesc>(
      name, ir::kOpTypeBatchNormalization, inputs, outputs, param);
  if (model_desc != nullptr) {
    model_desc->op_descs_.push_back(op_desc);
  }

  auto expr = std::make_shared<Expr>(op_desc);
  return expr;
}

// embedding
NNDEPLOY_CC_API std::shared_ptr<Expr> makeEmbedding(
    ir::ModelDesc *model_desc, std::shared_ptr<Expr> indices,
    std::string data_weight_name, std::string op_name,
    std::string output_name) {
  std::string name = op_name;
  if (name.empty()) {
    if (model_desc != nullptr) {
      int index = model_desc->op_descs_.size();
      name = "embedding" + std::to_string(index);
    } else {
      name = "embedding";
    }
  }
  std::vector<std::string> inputs = {indices->getOutputName()[0]};
  if (!data_weight_name.empty()) {
    inputs.push_back(data_weight_name);
  }
  std::vector<std::string> outputs;
  if (!output_name.empty()) {
    outputs.push_back(output_name);
  } else {
    outputs.push_back(name + ".output");
  }
  auto op_desc =
      std::make_shared<ir::OpDesc>(name, ir::kOpTypeEmbedding, inputs, outputs);
  if (model_desc != nullptr) {
    model_desc->op_descs_.push_back(op_desc);
  }

  auto expr = std::make_shared<Expr>(op_desc);
  return expr;
}

// gemm
NNDEPLOY_CC_API std::shared_ptr<Expr> makeGemm(
    ir::ModelDesc *model_desc, std::shared_ptr<Expr> input,
    std::shared_ptr<ir::GemmParam> param, const std::string &weight,
    const std::string &bias, std::string op_name, std::string output_name) {
  std::string name = op_name;
  if (name.empty()) {
    if (model_desc != nullptr) {
      int index = model_desc->op_descs_.size();
      name = "gemm" + std::to_string(index);
    } else {
      name = "gemm";
    }
  }
  std::vector<std::string> inputs = {input->getOutputName()[0]};
  if (!weight.empty()) {
    inputs.push_back(weight);
  }
  if (!bias.empty()) {
    inputs.push_back(bias);
  }
  std::vector<std::string> outputs;
  if (!output_name.empty()) {
    outputs.push_back(output_name);
  } else {
    outputs.push_back(name + ".output");
  }
  auto op_desc = std::make_shared<ir::OpDesc>(name, ir::kOpTypeGemm, inputs,
                                              outputs, param);
  if (model_desc != nullptr) {
    model_desc->op_descs_.push_back(op_desc);
  }
  auto expr = std::make_shared<Expr>(op_desc);
  return expr;
}

// matmul
NNDEPLOY_CC_API std::shared_ptr<Expr> makeMatmul(ir::ModelDesc *model_desc,
                                                 std::shared_ptr<Expr> input,
                                                 const std::string &weight,
                                                 const std::string &bias,
                                                 std::string op_name,
                                                 std::string output_name) {
  std::string name = op_name;
  if (name.empty()) {
    if (model_desc != nullptr) {
      int index = model_desc->op_descs_.size();
      name = "matmul" + std::to_string(index);
    } else {
      name = "matmul";
    }
  }
  std::vector<std::string> inputs = {input->getOutputName()[0]};
  NNDEPLOY_CHECK_PARAM_NULL_RET_NULL(weight.size(), "weight is null");
  inputs.push_back(weight);
  if (!bias.empty()) {
    inputs.push_back(bias);
  }
  std::vector<std::string> outputs;
  if (!output_name.empty()) {
    outputs.push_back(output_name);
  } else {
    outputs.push_back(name + ".output");
  }
  auto op_desc =
      std::make_shared<ir::OpDesc>(name, ir::kOpTypeMatMul, inputs, outputs);
  if (model_desc != nullptr) {
    model_desc->op_descs_.push_back(op_desc);
  }
  auto expr = std::make_shared<Expr>(op_desc);
  return expr;
}

NNDEPLOY_CC_API std::shared_ptr<Expr> makeMatmul(ir::ModelDesc *model_desc,
                                                 std::shared_ptr<Expr> input,
                                                 std::shared_ptr<Expr> weight,
                                                 const std::string &bias,
                                                 std::string op_name,
                                                 std::string output_name) {
  std::string name = op_name;
  if (name.empty()) {
    if (model_desc != nullptr) {
      int index = model_desc->op_descs_.size();
      name = "matmul" + std::to_string(index);
    } else {
      name = "matmul";
    }
  }
  std::vector<std::string> inputs = {input->getOutputName()[0],
                                     weight->getOutputName()[0]};
  if (!bias.empty()) {
    inputs.push_back(bias);
  }
  std::vector<std::string> outputs;
  if (!output_name.empty()) {
    outputs.push_back(output_name);
  } else {
    outputs.push_back(name + ".output");
  }
  auto op_desc =
      std::make_shared<ir::OpDesc>(name, ir::kOpTypeMatMul, inputs, outputs);
  if (model_desc != nullptr) {
    model_desc->op_descs_.push_back(op_desc);
  }
  auto expr = std::make_shared<Expr>(op_desc);
  return expr;
}

// flatten
NNDEPLOY_CC_API std::shared_ptr<Expr> makeFlatten(
    ir::ModelDesc *model_desc, std::shared_ptr<Expr> input,
    std::shared_ptr<ir::FlattenParam> param, std::string op_name,
    std::string output_name) {
  std::string name = op_name;
  if (name.empty()) {
    if (model_desc != nullptr) {
      int index = model_desc->op_descs_.size();
      name = "flatten" + std::to_string(index);
    } else {
      name = "flatten";
    }
  }
  std::vector<std::string> inputs = {input->getOutputName()[0]};
  std::vector<std::string> outputs;
  if (!output_name.empty()) {
    outputs.push_back(output_name);
  } else {
    outputs.push_back(name + ".output");
  }
  auto op_desc = std::make_shared<ir::OpDesc>(name, ir::kOpTypeFlatten, inputs,
                                              outputs, param);
  if (model_desc != nullptr) {
    model_desc->op_descs_.push_back(op_desc);
  }
  auto expr = std::make_shared<Expr>(op_desc);
  return expr;
}

// MaxPool
NNDEPLOY_CC_API std::shared_ptr<Expr> makeMaxPool(
    ir::ModelDesc *model_desc, std::shared_ptr<Expr> input,
    std::shared_ptr<ir::MaxPoolParam> param, std::string op_name,
    std::string output_name) {
  std::string name = op_name;
  if (name.empty()) {
    if (model_desc != nullptr) {
      int index = model_desc->op_descs_.size();
      name = "maxpool" + std::to_string(index);
    } else {
      name = "maxpool";
    }
  }
  std::vector<std::string> inputs = {input->getOutputName()[0]};
  std::vector<std::string> outputs;
  if (!output_name.empty()) {
    outputs.push_back(output_name);
  } else {
    outputs.push_back(name + ".output");
  }
  auto op_desc = std::make_shared<ir::OpDesc>(name, ir::kOpTypeMaxPool, inputs,
                                              outputs, param);
  if (model_desc != nullptr) {
    model_desc->op_descs_.push_back(op_desc);
  }
  auto expr = std::make_shared<Expr>(op_desc);
  return expr;
}

// GlobalAveragePool
NNDEPLOY_CC_API std::shared_ptr<Expr> makeGlobalAveragePool(
    ir::ModelDesc *model_desc, std::shared_ptr<Expr> input, std::string op_name,
    std::string output_name) {
  std::string name = op_name;
  if (name.empty()) {
    if (model_desc != nullptr) {
      int index = model_desc->op_descs_.size();
      name = "globalaveragepool" + std::to_string(index);
    } else {
      name = "globalaveragepool";
    }
  }
  std::vector<std::string> inputs = {input->getOutputName()[0]};
  std::vector<std::string> outputs;
  if (!output_name.empty()) {
    outputs.push_back(output_name);
  } else {
    outputs.push_back(name + ".output");
  }
  auto op_desc = std::make_shared<ir::OpDesc>(
      name, ir::kOpTypeGlobalAveragePool, inputs, outputs);
  if (model_desc != nullptr) {
    model_desc->op_descs_.push_back(op_desc);
  }
  auto expr = std::make_shared<Expr>(op_desc);
  return expr;
}

std::shared_ptr<Expr> makeRMSNorm(
    ir::ModelDesc *model_desc, std::shared_ptr<Expr> input,
    std::shared_ptr<ir::RMSNormParam> param, const std::string &weight,
    const std::string &residual, std::string op_name, std::string output_name) {
  std::string name = op_name;
  if (name.empty()) {
    if (model_desc != nullptr) {
      int index = model_desc->op_descs_.size();
      name = "rmsnorm" + std::to_string(index);
    } else {
      name = "rmsnorm";
    }
  }
  std::vector<std::string> inputs = {input->getOutputName()[0]};
  if (!weight.empty()) {
    inputs.push_back(weight);
  }
  if (!residual.empty()) {
    inputs.push_back(residual);
  }

  std::vector<std::string> outputs;
  if (!output_name.empty()) {
    outputs.push_back(output_name);
  } else {
    outputs.push_back(name + ".output");
  }
  auto op_desc = std::make_shared<ir::OpDesc>(name, ir::kOpTypeRMSNorm, inputs,
                                              outputs, param);
  if (model_desc != nullptr) {
    model_desc->op_descs_.push_back(op_desc);
  }

  auto expr = std::make_shared<Expr>(op_desc);
  return expr;
}

// cast
std::shared_ptr<Expr> makeCast(ir::ModelDesc *model_desc,
                               std::shared_ptr<Expr> input,
                               std::shared_ptr<ir::CastParam> param,
                               std::string op_name, std::string output_name) {
  std::string name = op_name;
  if (name.empty()) {
    if (model_desc != nullptr) {
      int index = model_desc->op_descs_.size();
      name = "cast" + std::to_string(index);
    } else {
      name = "cast";
    }
  }
  std::vector<std::string> inputs = {input->getOutputName()[0]};
  std::vector<std::string> outputs;
  if (!output_name.empty()) {
    outputs.push_back(output_name);
  } else {
    outputs.push_back(name + ".output");
  }
  auto op_desc = std::make_shared<ir::OpDesc>(name, ir::kOpTypeCast, inputs,
                                              outputs, param);
  if (model_desc != nullptr) {
    model_desc->op_descs_.push_back(op_desc);
  }

  auto expr = std::make_shared<Expr>(op_desc);
  return expr;
}

// reshape
std::shared_ptr<Expr> makeReshape(ir::ModelDesc *model_desc,
                                  std::shared_ptr<Expr> input,
                                  std::shared_ptr<Expr> new_shape,
                                  std::shared_ptr<ir::ReshapeParam> param,
                                  std::string op_name,
                                  std::string output_name) {
  std::string name = op_name;
  if (name.empty()) {
    if (model_desc != nullptr) {
      int index = model_desc->op_descs_.size();
      name = "reshape" + std::to_string(index);
    } else {
      name = "reshape";
    }
  }
  std::vector<std::string> inputs = {input->getOutputName()[0],
                                     new_shape->getOutputName()[0]};
  std::vector<std::string> outputs;
  if (!output_name.empty()) {
    outputs.push_back(output_name);
  } else {
    outputs.push_back(name + ".output");
  }
  auto op_desc = std::make_shared<ir::OpDesc>(name, ir::kOpTypeReshape, inputs,
                                              outputs, param);
  if (model_desc != nullptr) {
    model_desc->op_descs_.push_back(op_desc);
  }

  auto expr = std::make_shared<Expr>(op_desc);
  return expr;
}

std::shared_ptr<Expr> makeReshape(ir::ModelDesc *model_desc,
                                  std::shared_ptr<Expr> input,
                                  std::string weight_new_shape,
                                  std::shared_ptr<ir::ReshapeParam> param,
                                  std::string op_name,
                                  std::string output_name) {
  std::string name = op_name;
  if (name.empty()) {
    if (model_desc != nullptr) {
      int index = model_desc->op_descs_.size();
      name = "reshape" + std::to_string(index);
    } else {
      name = "reshape";
    }
  }
  std::vector<std::string> inputs = {input->getOutputName()[0],
                                     weight_new_shape};
  std::vector<std::string> outputs;
  if (!output_name.empty()) {
    outputs.push_back(output_name);
  } else {
    outputs.push_back(name + ".output");
  }
  auto op_desc = std::make_shared<ir::OpDesc>(name, ir::kOpTypeReshape, inputs,
                                              outputs, param);
  if (model_desc != nullptr) {
    model_desc->op_descs_.push_back(op_desc);
  }

  auto expr = std::make_shared<Expr>(op_desc);
  return expr;
}

// broadcast
std::shared_ptr<Expr> makeBroadcast(ir::ModelDesc *model_desc,
                                    std::shared_ptr<Expr> input,
                                    std::shared_ptr<Expr> broadcast_shape,
                                    std::string op_name,
                                    std::string output_name) {
  std::string name = op_name;
  if (name.empty()) {
    if (model_desc != nullptr) {
      int index = model_desc->op_descs_.size();
      name = "broadcast" + std::to_string(index);
    } else {
      name = "broadcast";
    }
  }
  std::vector<std::string> inputs = {input->getOutputName()[0]};
  std::vector<std::string> outputs;
  if (!output_name.empty()) {
    outputs.push_back(output_name);
  } else {
    outputs.push_back(name + ".output");
  }
  auto op_desc =
      std::make_shared<ir::OpDesc>(name, ir::kOpTypeBroadcast, inputs, outputs);
  if (model_desc != nullptr) {
    model_desc->op_descs_.push_back(op_desc);
  }

  auto expr = std::make_shared<Expr>(op_desc);
  return expr;
}

// transpose
std::shared_ptr<Expr> makeTranspose(ir::ModelDesc *model_desc,
                                    std::shared_ptr<Expr> input,
                                    std::shared_ptr<Expr> indices,
                                    std::string op_name,
                                    std::string output_name) {
  std::string name = op_name;
  if (name.empty()) {
    if (model_desc != nullptr) {
      int index = model_desc->op_descs_.size();
      name = "transpose" + std::to_string(index);
    } else {
      name = "transpose";
    }
  }
  std::vector<std::string> inputs = {input->getOutputName()[0],
                                     indices->getOutputName()[0]};
  std::vector<std::string> outputs;
  if (!output_name.empty()) {
    outputs.push_back(output_name);
  } else {
    outputs.push_back(name + ".output");
  }
  auto op_desc =
      std::make_shared<ir::OpDesc>(name, ir::kOpTypeTranspose, inputs, outputs);
  if (model_desc != nullptr) {
    model_desc->op_descs_.push_back(op_desc);
  }

  auto expr = std::make_shared<Expr>(op_desc);
  return expr;
}

std::shared_ptr<Expr> makeTranspose(ir::ModelDesc *model_desc,
                                    std::shared_ptr<Expr> input,
                                    const std::string &indices,
                                    std::string op_name,
                                    std::string output_name) {
  std::string name = op_name;
  if (name.empty()) {
    if (model_desc != nullptr) {
      int index = model_desc->op_descs_.size();
      name = "transpose" + std::to_string(index);
    } else {
      name = "transpose";
    }
  }
  NNDEPLOY_CHECK_PARAM_NULL_RET_NULL(indices.size(), "indices is null");
  std::vector<std::string> inputs = {input->getOutputName()[0], indices};
  std::vector<std::string> outputs;
  if (!output_name.empty()) {
    outputs.push_back(output_name);
  } else {
    outputs.push_back(name + ".output");
  }
  auto op_desc =
      std::make_shared<ir::OpDesc>(name, ir::kOpTypeTranspose, inputs, outputs);
  if (model_desc != nullptr) {
    model_desc->op_descs_.push_back(op_desc);
  }

  auto expr = std::make_shared<Expr>(op_desc);
  return expr;
}

// rotate_embedding
std::shared_ptr<Expr> makeRotateEmbedding(
    ir::ModelDesc *model_desc, std::shared_ptr<Expr> input,
    std::string inv_freq, std::string op_name,
    std::vector<std::string> output_names) {
  std::string name = op_name;
  if (name.empty()) {
    if (model_desc != nullptr) {
      int index = model_desc->op_descs_.size();
      name = "rotate_embedding" + std::to_string(index);
    } else {
      name = "rotate_embedding";
    }
  }
  std::vector<std::string> inputs = {input->getOutputName()[0]};
  inputs.emplace_back(inv_freq);
  std::vector<std::string> outputs;
  if (!output_names.empty()) {
    NNDEPLOY_ASSERT(output_names.size() == 2);
    for (auto name : output_names) {
      outputs.push_back(name);
    }
  } else {
    outputs.push_back(name + ".cos.output");
    outputs.push_back(name + ".sin.output");
  }
  auto op_desc =
      std::make_shared<ir::OpDesc>(name, ir::kOpTypeTranspose, inputs, outputs);
  if (model_desc != nullptr) {
    model_desc->op_descs_.push_back(op_desc);
  }

  auto expr = std::make_shared<Expr>(op_desc);
  return expr;
}
}  // namespace op
}  // namespace nndeploy
