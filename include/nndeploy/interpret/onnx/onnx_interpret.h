#ifndef _NNDEPLOY_INTERPRET_ONNX_INTERPRET_H_
#define _NNDEPLOY_INTERPRET_ONNX_INTERPRET_H_

#include "nndeploy/interpret/interpret.h"
#include "nndeploy/op/ir.h"
#include "nndeploy/op/op.h"
#include "onnx/common/assertions.h"
#include "onnx/common/file_utils.h"
#include "onnx/common/ir.h"
#include "onnx/common/ir_pb_converter.h"
#include "onnx/common/platform_helpers.h"
#include "onnx/onnx_pb.h"
#include "onnx/proto_utils.h"

namespace nndeploy {
namespace interpret {

class OnnxInterpret : public Interpret {
 public:
  OnnxInterpret();
  virtual ~OnnxInterpret();

  static base::DataType convertToDataType(
      const onnx::TensorProto_DataType &src);
  static base::IntVector convertToShape(const onnx::TensorShapeProto &src);
  static base::DataFormat convertToDataFormat(const onnx::TensorShapeProto &src,
                                              bool is_weight);
  static base::IntVector convertToShape(
      const google::protobuf::RepeatedField<::int64_t> &src);
  static base::DataFormat convertToDataFormat(
      const google::protobuf::RepeatedField<::int64_t> &src, bool is_weight);
  static std::shared_ptr<op::OpDesc> convertToOpDesc(
      const onnx::NodeProto &src);
  static device::Tensor *convertToTensor(const onnx::TensorProto &src);
  static std::shared_ptr<op::ValueDesc> convertToValueDesc(
      const onnx::ValueInfoProto &src);

  virtual base::Status interpret(const std::vector<std::string> &model_value,
                                 const std::vector<op::ValueDesc> &input);

 private:
  std::unique_ptr<onnx::ModelProto> onnx_model_;
};

class OnnxOpConvert {
 public:
  OnnxOpConvert() {}
  virtual ~OnnxOpConvert() {}

  virtual std::shared_ptr<op::OpDesc> convert(
      const onnx::NodeProto &onnx_node) = 0;

  base::Status convert(const onnx::NodeProto &onnx_node,
                       std::shared_ptr<op::OpDesc> &op_desc) {
    base::Status status = base::kStatusCodeOk;
    op_desc->name_ = onnx_node.name();
    std::vector<std::string> inputs;
    for (int i = 0; i < onnx_node.input_size(); ++i) {
      op_desc->inputs_.push_back(onnx_node.input(i));
    }
    for (int i = 0; i < onnx_node.output_size(); ++i) {
      op_desc->outputs_.push_back(onnx_node.output(i));
    }
    return status;
  };
};

/**
 * @brief 算子参数的转换类
 *s
 */
class OnnxOpConvertCreator {
 public:
  virtual ~OnnxOpConvertCreator(){};
  virtual std::shared_ptr<OnnxOpConvert> createOnnxOpConvert(
      const std::string &type) = 0;
};

/**
 * @brief 算子参数的转换类模板
 *
 * @tparam T
 */
template <typename T>
class TypeOnnxOpConvertCreator : public OnnxOpConvertCreator {
  virtual std::shared_ptr<OnnxOpConvert> createOnnxOpConvert(
      const std::string &type) {
    return std::make_shared<T>();
  }
};

/**
 * @brief Get the Global OnnxOpConvert Creator Map object
 *
 * @return std::map<op::OpType, std::shared_ptr<OnnxOpConvertCreator>>&
 */
std::map<std::string, std::shared_ptr<OnnxOpConvertCreator>>
    &getGlobalOnnxOpConvertCreatorMap();

/**
 * @brief 算子参数的创建类的注册类模板
 *
 * @tparam T
 */
template <typename T>
class TypeOnnxOpConvertRegister {
 public:
  explicit TypeOnnxOpConvertRegister(const std::string &type) {
    getGlobalOnnxOpConvertCreatorMap()[type] = std::shared_ptr<T>(new T());
  }
};

/**
 * @brief Create a OnnxOpConvert object
 *
 * @param type
 * @return std::shared_ptr<OnnxOpConvert>
 */
extern NNDEPLOY_CC_API std::shared_ptr<OnnxOpConvert> createOnnxOpConvert(
    const std::string &type);

#define REGISTER_ONNX_OP_CONVERT_IMPLEMENTION(op_type, onnx_op_convert_class) \
  TypeOnnxOpConvertRegister<TypeOnnxOpConvertCreator<onnx_op_convert_class>>  \
      g_##onnx_op_convert_class##_register(op_type);

}  // namespace interpret
}  // namespace nndeploy

#endif /* _NNDEPLOY_NET_ONNX_INTERPRET_H_ */
