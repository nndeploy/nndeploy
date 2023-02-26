
#ifndef _NNDEPLOY_INCLUDE_INTERPRET_INTERPRET_H_
#define _NNDEPLOY_INCLUDE_INTERPRET_INTERPRET_H_

#include "nndeploy/include/base/basic.h"
#include "nndeploy/include/base/log.h"
#include "nndeploy/include/base/macro.h"
#include "nndeploy/include/base/object.h"
#include "nndeploy/include/base/status.h"
#include "nndeploy/include/cryption/decrypt.h"
#include "nndeploy/include/interpret/abstract_interpret_impl.h"

namespace nndeploy {
namespace interpret {

/**
 * @brief 
 * # convert edit quantize forward 都要依赖这个模块
 */
class Interpret {
 public:
  Interpret();
  ~Interpret();

  base::Status init(base::InterpretConfig config, base::ShapeMap static_shape = base::ShapeMap());
  base::Status deinit();

  base::InterpretConfig getConfig();

  base::Status getStaticShape(base::ShapeMap shape_map);

  base::Status inferShape(base::ShapeMap static_shape);

  int getNumOfInput();
  int getNumOfOutput();

  std::vector<std::string> getInputNames();
  std::vector<std::string> getOutputNames();

  base::ShapeMap getInputShapeMap();
  base::ShapeMap getOutputShapeMap();

  std::shared_ptr<AbstractInterpretImpl> getAbstractInterpretImpl();
  std::shared_ptr<ir::Model> getModel();

 private:
  std::shared_ptr<AbstractInterpretImpl> abstract_interpret_impl_;
};

}  // namespace interpret
}  // namespace nndeploy

#endif