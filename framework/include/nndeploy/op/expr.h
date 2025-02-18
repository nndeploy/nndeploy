
#ifndef _NNDEPLOY_OP_EXPR_H_
#define _NNDEPLOY_OP_EXPR_H_

#include "nndeploy/base/macro.h"
#include "nndeploy/device/tensor.h"
#include "nndeploy/ir/ir.h"

namespace nndeploy {
namespace op {

// 标识Expr的类型
enum ExprType : int {
  // TODO：需要细分为输入、输出、权重吗？似乎现在这样更加简单且直观，但是只有这个信息能区分输入与权重吗？（应该可以？）
  kExprTypeValueDesc = 0x0000,
  kExprTypeOpDesc,
  kExprTypeModelDesc,
};

class NNDEPLOY_CC_API Expr {
 public:
  Expr(const std::string &name);
  Expr(const std::string &name, base::DataType data_type);
  Expr(const std::string &name, base::DataType data_type,
       base::IntVector shape);
  Expr(std::shared_ptr<ir::ValueDesc> value_desc);
  Expr(std::shared_ptr<ir::OpDesc> op_desc);
  Expr(std::shared_ptr<ir::ModelDesc> model_desc);

  ~Expr();

  std::vector<std::string> getOutputName();

 protected:
  ExprType expr_type_;
  std::shared_ptr<ir::ValueDesc> value_desc_;
  std::shared_ptr<ir::OpDesc> op_desc_;
  std::shared_ptr<ir::ModelDesc> model_desc_;
};

/**
 * @brief  一系列创建函数
 */
// input
NNDEPLOY_CC_API std::shared_ptr<Expr> makeInput(
    ir::ModelDesc *model_desc, std::string name,
    base::DataType data_type = base::dataTypeOf<float>(),
    base::IntVector shape = base::IntVector());
// output
NNDEPLOY_CC_API void makeOutput(ir::ModelDesc *model_desc,
                                std::shared_ptr<Expr> expr);
// block
NNDEPLOY_CC_API std::shared_ptr<Expr> makeBlock(
    ir::ModelDesc *model_desc, std::shared_ptr<ir::ModelDesc> model_block);
// conv2d
NNDEPLOY_CC_API std::shared_ptr<Expr> makeConv(
    ir::ModelDesc *model_desc, std::shared_ptr<Expr> input,
    std::shared_ptr<ir::ConvParam> param, const std::string &weight,
    const std::string &bias = "", std::string op_name = "",
    std::string output_name = "");
// relu
NNDEPLOY_CC_API std::shared_ptr<Expr> makeRelu(ir::ModelDesc *model_desc,
                                               std::shared_ptr<Expr> input,
                                               std::string op_name = "",
                                               std::string output_name = "");
// relu
NNDEPLOY_CC_API std::shared_ptr<Expr> makeSigmoid(ir::ModelDesc *model_desc,
                                                  std::shared_ptr<Expr> input,
                                                  std::string op_name = "",
                                                  std::string output_name = "");

// softmax
NNDEPLOY_CC_API std::shared_ptr<Expr> makeSoftMax(
    ir::ModelDesc *model_desc, std::shared_ptr<Expr> input,
    std::shared_ptr<ir::SoftmaxParam> param, std::string op_name = "",
    std::string output_name = "");

// batchnorm
NNDEPLOY_CC_API std::shared_ptr<Expr> makeBatchNorm(
    ir::ModelDesc *model_desc, std::shared_ptr<Expr> input,
    std::shared_ptr<ir::BatchNormalizationParam> param,
    const std::string &scale, const std::string &bias, const std::string &mean,
    const std::string &var, std::string op_name = "",
    std::string output_name = "");

// add
NNDEPLOY_CC_API std::shared_ptr<Expr> makeAdd(ir::ModelDesc *model_desc,
                                              std::shared_ptr<Expr> input_0,
                                              std::shared_ptr<Expr> input_1,
                                              std::string op_name = "",
                                              std::string output_name = "");

// mul
NNDEPLOY_CC_API std::shared_ptr<Expr> makeMul(ir::ModelDesc *model_desc,
                                              std::shared_ptr<Expr> input_0,
                                              std::shared_ptr<Expr> input_1,
                                              std::string op_name = "",
                                              std::string output_name = "");

// gemm
NNDEPLOY_CC_API std::shared_ptr<Expr> makeGemm(
    ir::ModelDesc *model_desc, std::shared_ptr<Expr> input,
    std::shared_ptr<ir::GemmParam> param, const std::string &weight,
    const std::string &bias = "", std::string op_name = "",
    std::string output_name = "");

// flatten
NNDEPLOY_CC_API std::shared_ptr<Expr> makeFlatten(
    ir::ModelDesc *model_desc, std::shared_ptr<Expr> input,
    std::shared_ptr<ir::FlattenParam> param, std::string op_name = "",
    std::string output_name = "");

// MaxPool
NNDEPLOY_CC_API std::shared_ptr<Expr> makeMaxPool(
    ir::ModelDesc *model_desc, std::shared_ptr<Expr> input,
    std::shared_ptr<ir::MaxPoolParam> param, std::string op_name = "",
    std::string output_name = "");

// GlobalAveragePool
NNDEPLOY_CC_API std::shared_ptr<Expr> makeGlobalAveragePool(
    ir::ModelDesc *model_desc, std::shared_ptr<Expr> input,
    std::string op_name = "", std::string output_name = "");

// TODO: @Leonisux:
// 补充llama的算子的手动构图函数
// embedding
NNDEPLOY_CC_API std::shared_ptr<Expr> makeEmbedding(
    ir::ModelDesc *model_desc, std::shared_ptr<Expr> indices,
    std::string op_name = "", std::string output_name = "");

// silu
NNDEPLOY_CC_API std::shared_ptr<Expr> makeSilu(ir::ModelDesc *model_desc,
                                               std::shared_ptr<Expr> input,
                                               std::string op_name = "",
                                               std::string output_name = "");

// matmul
NNDEPLOY_CC_API std::shared_ptr<Expr> makeMatmul(ir::ModelDesc *model_desc,
                                                 std::shared_ptr<Expr> input,
                                                 const std::string &weight,
                                                 const std::string &bias,
                                                 std::string op_name = "",
                                                 std::string output_name = "");
// matmul
NNDEPLOY_CC_API std::shared_ptr<Expr> makeMatmul(ir::ModelDesc *model_desc,
                                                 std::shared_ptr<Expr> input,
                                                 std::shared_ptr<Expr> weight,
                                                 const std::string &bias,
                                                 std::string op_name = "",
                                                 std::string output_name = "");

// rmsnorm
std::shared_ptr<Expr> makeRMSNorm(ir::ModelDesc *model_desc,
                                  std::shared_ptr<Expr> input,
                                  std::shared_ptr<ir::RMSNormParam> param,
                                  const std::string &weight,
                                  const std::string &residual,
                                  std::string op_name = "",
                                  std::string output_name = "");

// cast
std::shared_ptr<Expr> makeCast(
    ir::ModelDesc *model_desc, std::shared_ptr<Expr> input,
    std::shared_ptr<ir::CastParam> param =
        std::make_shared<ir::CastParam>(base::dataTypeOf<float>()),
    std::string op_name = "", std::string output_name = "");

// reshape
std::shared_ptr<Expr> makeReshape(ir::ModelDesc *model_desc,
                                  std::shared_ptr<Expr> input,
                                  std::shared_ptr<Expr> new_shape,
                                  std::shared_ptr<ir::ReshapeParam> param,
                                  std::string op_name = "",
                                  std::string output_name = "");

std::shared_ptr<Expr> makeReshape(ir::ModelDesc *model_desc,
                                  std::shared_ptr<Expr> input,
                                  std::string weight_new_shape,
                                  std::shared_ptr<ir::ReshapeParam> param,
                                  std::string op_name = "",
                                  std::string output_name = "");
// broadcast
std::shared_ptr<Expr> makeBroadcast(ir::ModelDesc *model_desc,
                                    std::shared_ptr<Expr> input,
                                    std::shared_ptr<Expr> broadcast_shape,
                                    std::string op_name = "",
                                    std::string output_name = "");
// transpose
std::shared_ptr<Expr> makeTranspose(ir::ModelDesc *model_desc,
                                    std::shared_ptr<Expr> input,
                                    std::shared_ptr<Expr> indices,
                                    std::string op_name = "",
                                    std::string output_name = "");

std::shared_ptr<Expr> makeTranspose(ir::ModelDesc *model_desc,
                                    std::shared_ptr<Expr> input,
                                    const std::string &indices,
                                    std::string op_name = "",
                                    std::string output_name = "");

// rotate embedding
std::shared_ptr<Expr> makeRotateEmbedding(
    ir::ModelDesc *model_desc, std::shared_ptr<Expr> input,
    std::string inv_freq, std::string op_name = "",
    std::vector<std::string> output_names = {});
}  // namespace op
}  // namespace nndeploy

#endif
