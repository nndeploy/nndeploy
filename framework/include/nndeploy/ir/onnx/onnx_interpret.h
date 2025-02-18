#ifndef _NNDEPLOY_IR_ONNX_ONNX_INTERPRET_H_
#define _NNDEPLOY_IR_ONNX_ONNX_INTERPRET_H_

#include "nndeploy/ir/interpret.h"
#include "nndeploy/ir/ir.h"
#include "onnx/common/assertions.h"
#include "onnx/common/file_utils.h"
#include "onnx/common/ir.h"
#include "onnx/common/ir_pb_converter.h"
#include "onnx/common/platform_helpers.h"
#include "onnx/onnx_pb.h"
#include "onnx/proto_utils.h"
#include "onnx/version_converter/convert.h"

namespace nndeploy {
namespace ir {

/**
 * @brief 使用的是最新的onnx版本 - target_ir_version=21
 * @link https://github.com/onnx/onnx/blob/main/docs/Versioning.md
 * @note：最好是能够在线升级到最新的onnx版本
 */
class OnnxInterpret : public Interpret {
 public:
  explicit OnnxInterpret(ModelDesc *model_desc = nullptr,
                         bool is_external = false);
  virtual ~OnnxInterpret();

  // convert
  static base::DataType convertToDataType(
      const onnx::TensorProto_DataType &src);
  static base::IntVector convertToShape(const onnx::TensorShapeProto &src);
  static base::DataFormat convertToDataFormat(const onnx::TensorShapeProto &src,
                                              bool is_weight);
  static base::IntVector convertToShape(
      const google::protobuf::RepeatedField<::int64_t> &src);
  static base::DataFormat convertToDataFormat(
      const google::protobuf::RepeatedField<::int64_t> &src, bool is_weight);
  static std::shared_ptr<OpDesc> convertToOpDesc(const onnx::NodeProto &src);
  static device::Tensor *convertToTensor(const onnx::TensorProto &src);
  static device::Tensor *convertToTensor(const onnx::TensorProto &src,
                                         const std::string &name);
  static std::shared_ptr<ValueDesc> convertToValueDesc(
      const onnx::ValueInfoProto &src);

  // get attribute value
  static onnx::AttributeProto_AttributeType getAttributeType(
      const char *type_name);
  static int32_t getAttributeInt(const onnx::NodeProto &node,
                                 const std::string &name, int default_value);
  static std::vector<int32_t> getAttributeIntVector(const onnx::NodeProto &node,
                                                    const std::string &name);
  static std::vector<int64_t> getAttributeInt64Vector(
      const onnx::NodeProto &node, const std::string &name);
  static float getAttributeFloat(const onnx::NodeProto &node,
                                 const std::string &name, float default_value);
  static std::string getAttributeString(const onnx::NodeProto &node,
                                        const std::string &name,
                                        std::string def);
  static std::vector<std::string> getAttributeStringVector(
      const onnx::NodeProto &node, const std::string &name);
  static std::vector<std::string> splitString(std::string &s,
                                              const std::string &c);
  static std::vector<uint8_t> getAttributeUInt8Vector(
      const onnx::NodeProto &node, const std::string &name);
  static std::vector<int8_t> asymmetric2Symmetric(
      std::vector<uint8_t> &raw_value, uint8_t zero_point);
  static onnx::TensorProto getAttributeTensor(const onnx::NodeProto &node,
                                              const char *key);
  static int getTensorProtoDataSize(const onnx::TensorProto &tp);
  static void *getDataFromTensor(const onnx::TensorProto &tensor);
  static const onnx::TensorProto *getTensorFromConstantNode(
      const onnx::NodeProto &constant_node);

  virtual base::Status interpret(
      const std::vector<std::string> &model_value,
      const std::vector<ValueDesc> &input = std::vector<ValueDesc>());

 private:
  int target_version_ = 20;
  std::unique_ptr<onnx::ModelProto> onnx_model_;
};

class OnnxOpConvert {
 public:
  OnnxOpConvert() {}
  virtual ~OnnxOpConvert() {}

  virtual std::shared_ptr<OpDesc> convert(const onnx::NodeProto &onnx_node) = 0;

  base::Status convert(const onnx::NodeProto &onnx_node,
                       std::shared_ptr<OpDesc> &op_desc) {
    base::Status status = base::kStatusCodeOk;
    op_desc->name_ = onnx_node.name();
    std::vector<std::string> inputs;
    for (int i = 0; i < onnx_node.input_size(); ++i) {
      op_desc->inputs_.push_back(onnx_node.input(i));
      // NNDEPLOY_LOGE("op_desc->inputs_ = %s\n", op_desc->inputs_[i].c_str());
    }
    for (int i = 0; i < onnx_node.output_size(); ++i) {
      op_desc->outputs_.push_back(onnx_node.output(i));
      // NNDEPLOY_LOGE("op_desc->outputs_ = %s\n",
      // op_desc->outputs_[i].c_str());
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
 * @return std::map<OpType, std::shared_ptr<OnnxOpConvertCreator>>&
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

}  // namespace ir
}  // namespace nndeploy

#endif /* _NNDEPLOY_NET_ONNX_INTERPRET_H_ */
