/**
 * @file abstract_model.h
 * @author your name (you@domain.com)
 * @brief
 * @version 0.1
 * @date 2022-11-26
 *
 * @copyright Copyright (c) 2022
 *
 */
#ifndef _NNKIT_INFERENCE_ABSTRACT_MODEL_
#define _NNKIT_INFERENCE_ABSTRACT_MODEL_

#include "nnkit/base/config.h"
#include "nnkit/base/status.h"
#include "nnkit/inference/abstract_session.h"


using namespace nnkit::base;
using namespace nnkit::device;

namespace nnkit {
namespace inference {

class Inference {
 public:
  explicit Model(InferenceConfig inference_config);

  virtual ~Model();

  virtual Status Init() = 0;
  Virtual Status SetConfig(const std::string &key, const Config &value) = 0;
  Virtual Status GetConfig(const std::string &key, Config &value) = 0;
  virtual Status Deinit() = 0;

  virtual Status GetStaticShape(ShapeMap shape_map);

  virtual std::shared_ptr<Session> CreateSession(
      InferenceConfig inference_config, ShapeMap min_shape = ShapeMap(),
      ShapeMap opt_shape = ShapeMap(), ShapeMap max_shape = ShapeMap());

 private:
  std::shared_ptr<AbstractModel> abstract_model_;
};

}  // namespace inference
}  // namespace nnkit

#endif