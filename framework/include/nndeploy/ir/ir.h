
#ifndef _NNDEPLOY_IR_IR_H_
#define _NNDEPLOY_IR_IR_H_

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
#include "nndeploy/ir/op_param.h"
#include "nndeploy/safetensors/safetensors.hh"

namespace nndeploy {
namespace ir {

/**
 * @brief 参照并扩充了onnx的格式，描述算子的基本信息
 * # 1. 算子名称
 * # 2. 算子类型
 * # 3. 算子输入
 * # 4. 算子输出
 * # 5. 算子的参数
 */
class NNDEPLOY_CC_API OpDesc {
 public:
  OpDesc();
  OpDesc(OpType op_type);
  OpDesc(const std::string &name, OpType op_type);
  OpDesc(const std::string &name, OpType op_type,
         std::shared_ptr<base::Param> op_param);
  OpDesc(const std::string &name, OpType op_type,
         std::initializer_list<std::string> inputs,
         std::initializer_list<std::string> outputs);
  OpDesc(const std::string &name, OpType op_type,
         std::initializer_list<std::string> inputs,
         std::initializer_list<std::string> outputs,
         std::shared_ptr<base::Param> op_param);
  OpDesc(const std::string &name, OpType op_type,
         std::vector<std::string> &inputs, std::vector<std::string> &outputs);
  OpDesc(const std::string &name, OpType op_type,
         std::vector<std::string> &inputs, std::vector<std::string> &outputs,
         std::shared_ptr<base::Param> op_param);

  virtual ~OpDesc();

  // 序列化
  base::Status serialize(std::ostream &stream) const;
  // 反序列化
  base::Status deserialize(const std::string &line);

 public:
  // 算子名称
  std::string name_;
  // 节点类型
  OpType op_type_;
  // 节点输入 : 包含 input、weight等所有参与计算的数据
  std::vector<std::string> inputs_;
  // 节点输出
  std::vector<std::string> outputs_;
  // 算子参数
  std::shared_ptr<base::Param> op_param_;
};

/**
 * @brief 参照onnx的格式，描述模型或者算子输入输出
 *
 */
class ValueDesc {
 public:
  ValueDesc();
  ValueDesc(const std::string &name);
  ValueDesc(const std::string &name, base::DataType data_type);
  ValueDesc(const std::string &name, base::DataType data_type,
            base::IntVector shape);

  virtual ~ValueDesc();

  // 序列化
  base::Status serialize(std::ostream &stream) const;
  // 反序列化
  base::Status deserialize(const std::string &line);

 public:
  // 名称
  std::string name_;
  // 数据类型
  base::DataType data_type_;
  // 张量形状
  base::IntVector shape_;
};

// onnx / 自定义模型 -》ir(interpret)

/**
 * @brief 参照onnx的格式，描述模型的结构
 *
 */
class ModelDesc {
 public:
  ModelDesc();
  virtual ~ModelDesc();

  base::Status dump(std::ostream &oss);

  // 序列化模型结构为文本
  base::Status serializeStructureToText(std::ostream &stream) const;
  // 反序列化文本为模型结构
  base::Status deserializeStructureFromText(
      std::istream &stream, const std::vector<ValueDesc> &input);
  // 序列化模型权重为二进制文件
  base::Status serializeWeightsToBinary(std::ostream &stream) const;
  // 序列化模型权重为safetensors
  base::Status serializeWeightsToSafetensors(safetensors::safetensors_t &safetensors) const;
  base::Status serializeWeightsToSafetensorsImpl(safetensors::safetensors_t &safetensors, bool serialize_buffer = false) const;
  // 从二进制文件反序列化模型权重
  base::Status deserializeWeightsFromBinary(std::istream &stream);

 public:
  // 描述模型的名称
  std::string name_;
  // 模型输入
  std::vector<std::shared_ptr<ValueDesc>> inputs_;
  // 模型输出
  std::vector<std::shared_ptr<ValueDesc>> outputs_;
  // 模型算子列表
  std::vector<std::shared_ptr<OpDesc>> op_descs_;
  // 模型权重
  std::map<std::string, device::Tensor *> weights_;
  // 模型中间值，一般通常为空，多用于调试
  std::vector<std::shared_ptr<ValueDesc>> values_;
  // 模型块
  std::vector<std::shared_ptr<ModelDesc>> blocks_;
};

}  // namespace ir
}  // namespace nndeploy

#endif /* _NNDEPLOY_IR_IR_H_ */
