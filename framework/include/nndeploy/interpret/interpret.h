
#ifndef _NNDEPLOY_INTERPRET_INTERPRET_H_
#define _NNDEPLOY_INTERPRET_INTERPRET_H_

#include "nndeploy/op/ir.h"
#include "nndeploy/op/op.h"

namespace nndeploy {
namespace interpret {

class Interpret {
 public:
  /**
   * @brief Interpret类的默认构造函数
   *
   * 创建一个新的Interpret对象，并初始化model_desc_成员。
   * model_desc_被初始化为一个新的op::ModelDesc对象。
   */
  Interpret() { model_desc_ = new op::ModelDesc(); };
  /**
   * @brief 虚析构函数
   *
   * 负责清理Interpret对象，释放model_desc_指针指向的内存
   */
  virtual ~Interpret() {
    if (model_desc_ != nullptr) {
      delete model_desc_;
    }
  };

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
   * @see op::ValueDesc 了解输入描述的详细信息
   * @see base::Status 了解可能的返回状态
   */
  virtual base::Status interpret(const std::vector<std::string> &model_value,
                                 const std::vector<op::ValueDesc> &input =
                                     std::vector<op::ValueDesc>()) = 0;

  /**
   * @brief 获取模型描述
   *
   * 获取模型描述
   * @return 模型描述
   */
  op::ModelDesc *getModelDesc() { return model_desc_; };

 protected:
  /**
   * @brief 模型描述
   *
   * 用于存储模型描述信息
   */
  op::ModelDesc *model_desc_;
};

}  // namespace interpret
}  // namespace nndeploy

#endif /* _NNDEPLOY_INTERPRET_INTERPRET_H_ */