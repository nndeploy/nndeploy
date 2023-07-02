
#ifndef _NNDEPLOY_INCLUDE_INTERPRET_ABSTRACT_INTERPRET_IMPL_H_
#define _NNDEPLOY_INCLUDE_INTERPRET_ABSTRACT_INTERPRET_IMPL_H_

#include "nndeploy/include/base/basic.h"
#include "nndeploy/include/base/log.h"
#include "nndeploy/include/base/macro.h"
#include "nndeploy/include/base/object.h"
#include "nndeploy/include/base/status.h"
#include "nndeploy/include/cryption/decrypt.h"

namespace nndeploy {
namespace interpret {

/**
 * @brief
 * # convert edit quantize forward 都要依赖这个模块
 */
class AbstractInterpretImpl {
 public:
  AbstractInterpretImpl();
  ~AbstractInterpretImpl();

  virtual base::Status init(InterpretConfig config,
                            base::ShapeMap static_shape = base::ShapeMap()) = 0;
  virtual base::Status deinit() = 0;

  InterpretConfig getConfig();

  base::Status getStaticShape(base::ShapeMap shape_map);

  // 委托给ir::Model
  base::Status inferShape(base::ShapeMap static_shape);

  // 委托给ir::Model
  int getNumOfInput();
  int getNumOfOutput();

  // 委托给ir::Model
  std::vector<std::string> getInputNames();
  std::vector<std::string> getOutputNames();

  // 委托给ir::Model
  base::ShapeMap getInputShapeMap();
  base::ShapeMap inferShapeMap();

  std::shared_ptr<ir::Model> getModel();

 protected:
  InterpretConfig config_;
  base::ShapeMap static_shape_ = base::ShapeMap();

  std::shared_ptr<cryption::Decrypt> decrypt_;

  /**
   * @brief
   * # 自定义模型文件 -> ir::Model
   * # onnx模型文件 -> ir::Model
   */
  std::shared_ptr<ir::Model> model_;
};

}  // namespace interpret
}  // namespace nndeploy

#endif