#ifndef _NNDEPLOY_OP_MODEL_DESC_H_
#define _NNDEPLOY_OP_MODEL_DESC_H_

#include "nndeploy/base/common.h"
#include "nndeploy/base/glic_stl_include.h"
#include "nndeploy/base/log.h"
#include "nndeploy/base/macro.h"
#include "nndeploy/base/object.h"
#include "nndeploy/base/param.h"
#include "nndeploy/base/status.h"
#include "nndeploy/base/string.h"
#include "nndeploy/base/value.h"
#include "nndeploy/device/tensor.h"
#include "nndeploy/op/base/conv2d.h"
#include "nndeploy/op/expr.h"
#include "nndeploy/op/ir.h"

namespace nndeploy {

namespace op {

/**
 * @brief 参照onnx的格式，描述模型的结构
 *
 */
class ModelDesc {
 public:
  ModelDesc(){};
  virtual ~ModelDesc(){};

 public:
  /**
   * @brief  一系列创建函数
   */
  // conv2d
  std::shared_ptr<Expr> MakeConv2d(std::shared_ptr<Expr> input,
                                   std::shared_ptr<Expr> weight,
                                   std::shared_ptr<Conv2dParam> param,
                                   std::string name);
  // relu
  std::shared_ptr<Expr> MakeRelu(std::shared_ptr<Expr> input, std::string name);

 public:
  // 描述模型的名称
  std::string name_;
  // 模型算子列表
  std::vector<std::shared_ptr<OpDescAndParam>> op_desc_params_;
  // 模型权重
  std::map<std::string, device::Tensor *> weights_;
  // 模型输入
  std::vector<ValueDesc *> inputs_;
  // 模型输出
  std::vector<ValueDesc *> outputs_;
  // 模型中间值，一般通常为空，多用于调试
  std::vector<ValueDesc *> values_;
};

}  // namespace op
}  // namespace nndeploy
#endif