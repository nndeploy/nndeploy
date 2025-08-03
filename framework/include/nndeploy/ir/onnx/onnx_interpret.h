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
 * @brief 使用的是最新的onnx版本 - target_ir_version=20
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

  // 获取ONNX节点的各种属性值,包括整型、浮点型、字符串等
  // 根据类型名称获取ONNX属性类型
  static onnx::AttributeProto_AttributeType getAttributeType(
      const char *type_name);
  // 获取ONNX节点的整型属性值,如果不存在则返回默认值
  static int32_t getAttributeInt(const onnx::NodeProto &node,
                                 const std::string &name, int default_value);
  // 获取ONNX节点的整型数组属性值,返回int32_t类型的vector
  static std::vector<int32_t> getAttributeIntVector(const onnx::NodeProto &node,
                                                    const std::string &name);
  // 获取ONNX节点的64位整型数组属性值,返回int64_t类型的vector
  static std::vector<int64_t> getAttributeInt64Vector(
      const onnx::NodeProto &node, const std::string &name);
  // 获取ONNX节点的浮点型属性值,如果不存在则返回默认值
  static float getAttributeFloat(const onnx::NodeProto &node,
                                 const std::string &name, float default_value);
  // 获取ONNX节点的字符串属性值,如果不存在则返回默认字符串
  static std::string getAttributeString(const onnx::NodeProto &node,
                                        const std::string &name,
                                        std::string def);
  // 获取ONNX节点的字符串数组属性值,返回string类型的vector
  static std::vector<std::string> getAttributeStringVector(
      const onnx::NodeProto &node, const std::string &name);
  // 按指定分隔符分割字符串,返回分割后的字符串数组
  static std::vector<std::string> splitString(std::string &s,
                                              const std::string &c);
  // 获取ONNX节点的无符号8位整型数组属性值,返回uint8_t类型的vector
  static std::vector<uint8_t> getAttributeUInt8Vector(
      const onnx::NodeProto &node, const std::string &name);
  // 将非对称量化的uint8数据转换为对称量化的int8数据,需要指定零点
  static std::vector<int8_t> asymmetric2Symmetric(
      std::vector<uint8_t> &raw_value, uint8_t zero_point);
  // 获取ONNX节点的张量属性值,返回TensorProto对象
  static onnx::TensorProto getAttributeTensor(const onnx::NodeProto &node,
                                              const char *key);

  // 获取张量数据的总大小(元素个数)
  static int getTensorProtoDataSize(const onnx::TensorProto &tp);

  // 从张量中获取原始数据指针
  static void *getDataFromTensor(const onnx::TensorProto &tensor);

  // 从常量节点中获取张量
  static const onnx::TensorProto *getTensorFromConstantNode(
      const onnx::NodeProto &constant_node);

  virtual base::Status interpret(
      const std::vector<std::string> &model_value,
      const std::vector<ValueDesc> &input = std::vector<ValueDesc>());

 private:
  int target_version_ = 20;
  std::unique_ptr<onnx::ModelProto> onnx_model_;
};

/**
 * @brief ONNX算子转换基类
 * 用于将ONNX算子转换为IR中的OpDesc
 */
class OnnxOpConvert {
 public:
  OnnxOpConvert() {}
  virtual ~OnnxOpConvert() {}

  /**
   * @brief
   * 这个函数接收ONNX节点，返回转换后的OpDesc对象。绝大部分Op（如卷积、池化等）都需要单独的子类来实现这个函数
   * @param onnx_node ONNX节点
   * @return 转换后的OpDesc智能指针
   */
  virtual std::shared_ptr<OpDesc> convert(const onnx::NodeProto &onnx_node) = 0;

  /**
   * @brief 通用转换函数（所有子类共享）
   * @param onnx_node ONNX节点
   * @param op_desc 待填充的OpDesc
   * @return 状态码
   * @note
   * 这个函数处理所有操作都有的共同属性：
   *   1. 复制操作名称
   *   2. 复制输入节点列表
   *   3. 复制输出节点列表
   * 不需要每个子类重复编写这些共同逻辑，提高了代码复用性。
   */
  base::Status convert(const onnx::NodeProto &onnx_node,
                       std::shared_ptr<OpDesc> &op_desc) {
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
  virtual ~OnnxOpConvertCreator() {};
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
std::map<std::string, std::shared_ptr<OnnxOpConvertCreator>> &
getGlobalOnnxOpConvertCreatorMap();

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
