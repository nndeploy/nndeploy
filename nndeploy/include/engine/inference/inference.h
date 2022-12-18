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
#ifndef _NNDEPLOY_INCLUDE_INFERENCE_INFERENCE_
#define _NNDEPLOY_INCLUDE_INFERENCE_INFERENCE_

#include "nndeploy/include/base/config.h"
#include "nndeploy/include/base/status.h"
#include "nndeploy/include/inference/abstract_session.h"


using namespace nndeploy::base;
using namespace nndeploy::device;

namespace nndeploy {
namespace inference {

class InferenceNode : Node {
 public:
  explicit Inference(InferenceConfig inference_config);

  virtual ~Inference();

  virtual Status Init() = 0;
  Virtual Status SetConfig(const std::string &key, const Config &value) = 0;
  Virtual Status GetConfig(const std::string &key, Config &value) = 0;
  virtual Status Deinit() = 0;

  virtual Predict(TensorMap input_tensors, TensorMap output_tensor)

 private:
  std::shared_ptr<Model> model_;
  std::shared_ptr<Model> session_;
};

}  // namespace inference
}  // namespace nndeploy

#endif