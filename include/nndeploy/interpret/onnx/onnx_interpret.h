#ifndef _NNDEPLOY_INTERPRET_ONNX_INTERPRET_H_
#define _NNDEPLOY_INTERPRET_ONNX_INTERPRET_H_

#include "nndeploy/interpret/interpret.h"
#include "nndeploy/op/ir.h"
#include "nndeploy/op/op.h"
#include "onnx/onnx_pb.h"

namespace nndeploy {
namespace interpret {

class OnnxInterpret : public Interpret {
 public:
  OnnxInterpret() : Interpret(){};
  virtual ~OnnxInterpret(){};

  base::DataType convertToDataType(onnx::TensorProto::DataType data_type);
  base::IntVector convertToShape(onnx::TensorShapeProto_Dimension dim);

  std::shared_ptr<op::OpDesc> convertToOpDesc(const onnx::NodeProto &onnx_node);
  device::Tensor *convertToTensor(const onnx::TensorProto &initializer);
  std::shared_ptr<op::ValueDesc> convertToValueDesc(
      const onnx::ValueInfoProto &input);

  virtual base::Status interpret(const std::vector<std::string> &model_value,
                                 const std::vector<op::ValueDesc> &input);

 private:
  std::unique_ptr<onnx::ModelProto> onnx_model_;
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

#define REGISTER_OP_PARAM_IMPLEMENTION(op_type, op_param_class) \
  TypeOpParamRegister<TypeOpParamCreator<op_param_class>>       \
      g_##op_type##_##op_param_class##_register(op_type);

}  // namespace interpret
}  // namespace nndeploy

#endif /* _NNDEPLOY_NET_ONNX_INTERPRET_H_ */
