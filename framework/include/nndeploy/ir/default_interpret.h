
#ifndef _NNDEPLOY_IR_DEFAULT_INTERPRET_H_
#define _NNDEPLOY_IR_DEFAULT_INTERPRET_H_

#include "nndeploy/ir/interpret.h"
#include "nndeploy/ir/ir.h"

namespace nndeploy {
namespace ir {

class NNDEPLOY_CC_API DefaultInterpret : public Interpret {
 public:
  /**
   * @brief DefaultInterpret类的默认构造函数
   *
   * 创建一个新的DefaultInterpret对象，并初始化model_desc_成员。
   * model_desc_被初始化为一个新的ModelDesc对象。
   */
  // DefaultInterpret();

  /**
   * @brief DefaultInterpret类的带参构造函数
   *
   * 使用已有的model_desc创建一个新的 DefaultInterpret对象。
   */
  explicit DefaultInterpret(ModelDesc *model_desc = nullptr,
                            bool is_external = false);

  /**
   * @brief 虚析构函数
   *
   * 负责清理DefaultInterpret对象，释放model_desc_指针指向的内存
   */
  virtual ~DefaultInterpret();

  virtual base::Status interpret(
      const std::vector<std::string> &model_value,
      const std::vector<ValueDesc> &input = std::vector<ValueDesc>());

 private:
  /**
   * @brief 模型权重
   *
   * 用于存储模型权重信息
   */
  std::vector<std::shared_ptr<safetensors::safetensors_t>> st_ptr_;
};

}  // namespace ir
}  // namespace nndeploy

#endif /* _NNDEPLOY_IR_DEFAULT_INTERPRET_H_ */
