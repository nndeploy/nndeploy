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
#ifndef _NN_DEPLOY_INFERENCE_INFERENCE_
#define _NN_DEPLOY_INFERENCE_INFERENCE_

#include "nn_deploy/base/config.h"
#include "nn_deploy/base/status.h"
#include "nn_deploy/inference/abstract_session.h"


using namespace nn_deploy::base;
using namespace nn_deploy::device;

namespace nn_deploy {
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
}  // namespace nn_deploy

#endif