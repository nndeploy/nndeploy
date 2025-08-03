
#ifndef _NNDEPLOY_IR_INTERPRET_H_
#define _NNDEPLOY_IR_INTERPRET_H_

#include "nndeploy/ir/ir.h"

namespace nndeploy {
namespace ir {

class NNDEPLOY_CC_API Interpret {
 public:
  /**
   * @brief 解释器类的默认构造函数
   *
   * 创建一个新的Interpret对象，并初始化model_desc_成员。
   * model_desc_被初始化为一个新的ModelDesc对象。
   */
  // Interpret();

  /**
   * @brief 解释器类的带参构造函数
   *
   * 使用已有的model_desc创建一个新的Interpret对象。
   */
  explicit Interpret(ModelDesc *model_desc = nullptr, bool is_external = false);

  /**
   * @brief 虚析构函数
   *
   * 负责清理Interpret对象，释放model_desc_指针指向的内存
   */
  virtual ~Interpret();

  /**
   * @brief 解释模型
   *
   * 该函数负责解释输入的模型，将其转换为内部的中间表示（IR）格式。
   * 这个过程通常包括解析模型结构、提取参数、分析计算图等步骤。
   *
   * @param model_value
   * 包含模型信息的字符串向量。可能包含模型文件路径、序列化的模型数据等。
   * @param input 模型输入的描述信息。默认为空向量，表示使用模型的默认输入配置。
   *              当需要自定义输入时，可以通过此参数指定。
   *
   * @return base::Status 返回解释过程的状态。
   *         - 如果解释成功，返回 base::kStatusCodeOk
   *         - 如果解释失败，返回对应的错误状态码
   *
   * @note 这是一个纯虚函数，需要在派生类中实现具体的解释逻辑。
   *       不同的模型格式（如ONNX、TensorFlow等）可能需要不同的解释实现。
   *
   * @see ValueDesc 了解输入描述的详细信息
   * @see base::Status 了解可能的返回状态
   */
  virtual base::Status interpret(
      const std::vector<std::string> &model_value,
      const std::vector<ValueDesc> &input = std::vector<ValueDesc>()) = 0;

  /**
   * @brief
   *
   */
  // 打印模型结构
  base::Status dump(std::ostream &oss = std::cout);

  /**
   * @brief 存储模型结构以及模型权重
   *
   * 该函数负责将模型的结构和权重存储到指定的输出流中。
   * 这个过程通常包括序列化模型结构、序列化模型权重等步骤。
   *
   * @param structure_stream
   * 输出流，用于存储模型结构的序列化数据。
   * @param weight_file_path
   * 输出流，用于存储模型权重的序列化数据。
   *
   * @return base::Status 返回存储过程的状态。
   *         - 如果存储成功，返回 base::kStatusCodeOk
   *         - 如果存储失败，返回对应的错误状态码
   *
   * @note 这是一个虚函数，可以在派生类中重载以实现特定的存储逻辑。
   *
   * @see base::Status 了解可能的返回状态
   */
  base::Status saveModel(std::string &structure_str,
                         std::shared_ptr<safetensors::safetensors_t> st_ptr);
  /**
   * @brief 存储模型结构以及模型权重到指定路径
   *
   * 该函数负责将模型的结构和权重存储到指定的输出流中。
   * 这个过程通常包括序列化模型结构、序列化模型权重等步骤。
   *
   * @param structure_file_path
   * 输出流，用于存储模型结构的序列化数据。
   * @param weight_file_path
   * 输出流，用于存储模型权重的序列化数据。
   *
   * @return base::Status 返回存储过程的状态。
   *         - 如果存储成功，返回 base::kStatusCodeOk
   *         - 如果存储失败，返回对应的错误状态码
   *
   * @note 这是一个虚函数，可以在派生类中重载以实现特定的存储逻辑。
   *
   * @see base::Status 了解可能的返回状态
   */
  base::Status saveModelToFile(const std::string &structure_file_path,
                               const std::string &weight_file_path);

  /**
   * @brief 获取模型描述
   *
   * 获取模型描述
   * @return 模型描述
   */
  ModelDesc *getModelDesc();

 public:
  /**
   * @brief 模型描述
   *
   * 用于存储模型描述信息
   */
  ModelDesc *model_desc_ = nullptr;

  /**
   * @brief 是否是外部模型
   */
  bool is_external_ = false;
};

/**
 * @brief 解释器的创建类
 *
 */
class InterpretCreator {
 public:
  virtual ~InterpretCreator() {};
  // virtual Interpret *createInterpret(base::ModelType type) = 0;
  virtual Interpret *createInterpret(base::ModelType type,
                                     ir::ModelDesc *model_desc = nullptr,
                                     bool is_external = false) = 0;
  virtual std::shared_ptr<Interpret> createInterpretSharedPtr(
      base::ModelType type, ir::ModelDesc *model_desc = nullptr,
      bool is_external = false) = 0;
};

/**
 * @brief 解释器的创建类模板
 *
 * @tparam T
 */
template <typename T>
class TypeInterpretCreator : public InterpretCreator {
  virtual Interpret *createInterpret(base::ModelType type,
                                     ir::ModelDesc *model_desc = nullptr,
                                     bool is_external = false) {
    return new T(model_desc, is_external);
  }
  virtual std::shared_ptr<Interpret> createInterpretSharedPtr(
      base::ModelType type, ir::ModelDesc *model_desc = nullptr,
      bool is_external = false) {
    return std::make_shared<T>(model_desc, is_external);
  }
};

/**
 * @brief Get the Global Interpret Creator Map object
 *
 * @return std::map<base::ModelType, std::shared_ptr<InterpretCreator>>&
 */
extern NNDEPLOY_CC_API std::map<base::ModelType,
                                 std::shared_ptr<InterpretCreator>> &
    getGlobalInterpretCreatorMap();

/**
 * @brief 解释器的创建类的注册类模板
 *
 * @tparam T
 */
template <typename T>
class TypeInterpretRegister {
 public:
  explicit TypeInterpretRegister(base::ModelType type) {
    getGlobalInterpretCreatorMap()[type] = std::shared_ptr<T>(new T());
  }
};

/**
 * @brief Create a Interpret object
 *
 * @param type
 * @return Interpret*
 */
extern NNDEPLOY_CC_API Interpret *createInterpret(
    base::ModelType type, ir::ModelDesc *model_desc = nullptr,
    bool is_external = false);

extern NNDEPLOY_CC_API std::shared_ptr<Interpret> createInterpretSharedPtr(
    base::ModelType type, ir::ModelDesc *model_desc = nullptr,
    bool is_external = false);

}  // namespace ir
}  // namespace nndeploy

#endif /* _NNDEPLOY_IR_INTERPRET_H_ */