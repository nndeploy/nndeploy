
#ifndef _NNDEPLOY_IR_DEFAULT_INTERPRET_H_
#define _NNDEPLOY_IR_DEFAULT_INTERPRET_H_

#include "nndeploy/ir/interpret.h"
#include "nndeploy/ir/ir.h"

namespace nndeploy {
namespace ir {

class DefaultInterpret : public Interpret {
 public:
  /**
   * @brief DefaultInterpret类的默认构造函数
   *
   * 创建一个新的DefaultInterpret对象，并初始化model_desc_成员。
   * model_desc_被初始化为一个新的ModelDesc对象。
   */
  DefaultInterpret();
  /**
   * @brief 虚析构函数
   *
   * 负责清理DefaultInterpret对象，释放model_desc_指针指向的内存
   */
  virtual ~DefaultInterpret();

  virtual base::Status interpret(
      const std::vector<std::string> &model_value,
      const std::vector<ValueDesc> &input = std::vector<ValueDesc>());

  // virtual base::Status DefaultInterpret::interpret_safetensors(
  //     const std::string &weights_name);
};

}  // namespace ir
}  // namespace nndeploy

#endif /* _NNDEPLOY_IR_DEFAULT_INTERPRET_H_ */
