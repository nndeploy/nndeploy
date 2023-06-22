
#ifndef _NNDEPLOY_SOURCE_INTERPRET_INTERPRET_H_
#define _NNDEPLOY_SOURCE_INTERPRET_INTERPRET_H_

#include "nndeploy/source/base/basic.h"
#include "nndeploy/source/base/log.h"
#include "nndeploy/source/base/macro.h"
#include "nndeploy/source/base/object.h"
#include "nndeploy/source/base/status.h"
#include "nndeploy/source/cryption/decrypt.h"
#include "nndeploy/source/interpret/abstract_interpret_impl.h"

namespace nndeploy {
namespace interpret {

class InterpretConfig {
 public:
  base::ModelType model_type_;
  bool is_path_ = true;
  std::vector<std::string> model_value_;
  EncryptType is_encrypt_ = kEncryptTypeNone;
  std::string license_;
}

/**
 * @brief
 * # convert edit quantize forward 都要依赖这个模块
 */
class Interpret {
 public:
  Interpret();
  ~Interpret();

  base::Status init(InterpretConfig config,
                    base::ShapeMap static_shape = base::ShapeMap());
  base::Status deinit();

  base::InterpretConfig getConfig();

  base::Status getStaticShape(base::ShapeMap shape_map);

  base::Status inferShape(base::ShapeMap static_shape);

  int getNumOfInput();
  int getNumOfOutput();

  std::vector<std::string> getInputNames();
  std::vector<std::string> getOutputNames();

  base::ShapeMap getInputShapeMap();
  base::ShapeMap inferShapeMap();

  std::shared_ptr<AbstractInterpretImpl> getAbstractInterpretImpl();
  std::shared_ptr<ir::Model> getModel();

 private:
  std::shared_ptr<AbstractInterpretImpl> abstract_interpret_impl_;
};

}  // namespace interpret
}  // namespace nndeploy

#endif