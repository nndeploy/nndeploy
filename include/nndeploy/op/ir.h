
#ifndef _NNDEPLOY_OP_IR_H_
#define _NNDEPLOY_OP_IR_H_

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

namespace nndeploy {
namespace op {

enum OpType : int {
  kOpTypeForward = 0x0000,

  // unary
  kOpTypeRelu,

  // binary
  kOpTypeAdd,

  // shape

  // computation intensive
  kOpTypeConv2d,

  kOpTypeNone,
};

/**
 * @brief 参照并扩充了onnx的格式，描述算子的基本信息
 * # 1. 算子名称
 * # 2. 算子类型
 * # 3. 算子输入
 * # 4. 算子输出
 * # 5. 算子权重 - 将权重单独列出
 */
class NNDEPLOY_CC_API OpDesc {
 public:
  OpDesc() {}

  OpDesc(const std::string &name, OpType op_type)
      : name_(name), op_type_(op_type) {}

  OpDesc(const std::string &name, OpType op_type,
         std::initializer_list<std::string> inputs,
         std::initializer_list<std::string> outputs)
      : name_(name), op_type_(op_type), inputs_(inputs), outputs_(outputs) {}

  OpDesc(const std::string &name, OpType op_type,
         std::vector<std::string> &inputs, std::vector<std::string> &outputs)
      : name_(name), op_type_(op_type), inputs_(inputs), outputs_(outputs) {}

  virtual ~OpDesc() {}

  // 算子名称
  std::string name_;
  // 节点类型
  OpType op_type_;
  // 节点输入 : 包含 input、weight等所有参与计算的数据
  std::vector<std::string> inputs_;
  // 节点输出
  std::vector<std::string> outputs_;
};

/**
 * @brief
 *
 */
class NNDEPLOY_CC_API OpDescAndParam {
 public:
  OpDescAndParam() {}
  virtual ~OpDescAndParam() {}

  // 算子描述
  OpDesc op_desc_;
  // 算子参数
  std::shared_ptr<base::Param> op_param_;
};

/**
 * @brief 参照onnx的格式，描述模型或者算子输入输出
 *
 */
class ValueDesc {
 public:
  // 名称
  std::string name_;
  // 数据类型
  base::DataType data_type_;
  // 张量形状
  base::IntVector shape_;
};

/**
 * @brief 算子参数的创建类
 *
 */
class OpParamCreator {
 public:
  virtual ~OpParamCreator(){};
  virtual std::shared_ptr<base::Param> createOpParam(OpType type) = 0;
};

/**
 * @brief 算子参数的创建类模板
 *
 * @tparam T
 */
template <typename T>
class TypeOpParamCreator : public OpParamCreator {
  virtual std::shared_ptr<base::Param> createOpParam(OpType type) {
    return std::make_shared<T>();
  }
};

/**
 * @brief Get the Global base::Param Creator Map object
 *
 * @return std::map<OpType, std::shared_ptr<OpParamCreator>>&
 */
std::map<OpType, std::shared_ptr<OpParamCreator>> &getGlobalOpParamCreatorMap();

/**
 * @brief 算子参数的创建类的注册类模板
 *
 * @tparam T
 */
template <typename T>
class TypeOpParamRegister {
 public:
  explicit TypeOpParamRegister(OpType type) {
    getGlobalOpParamCreatorMap()[type] = std::shared_ptr<T>(new T());
  }
};

/**
 * @brief Create a base::Param object
 *
 * @param type
 * @return std::shared_ptr<base::Param>
 */
extern NNDEPLOY_CC_API std::shared_ptr<base::Param> createOpParam(
    OpType op_type);

}  // namespace op
}  // namespace nndeploy

#endif /* _NNDEPLOY_OP_IR_H_ */