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
#ifndef _NNDEPLOY_INCLUDE_INFERENCE_FORWARD_H_
#define _NNDEPLOY_INCLUDE_INFERENCE_FORWARD_H_

#include "nndeploy/include/base/log.h"
#include "nndeploy/include/base/macro.h"
#include "nndeploy/include/base/object.h"
#include "nndeploy/include/base/status.h"
#include "nndeploy/include/base/basic.h"
#include "nndeploy/include/device/device.h"
#include "nndeploy/include/inference/config.h"
#include "nndeploy/include/inference/forward_impl.h"
#include "nndeploy/include/inference/tensor.h"

namespace nndeploy {
namespace inference {

class Forward {
 public:
  Forward(Config config);

  ~Forward();

  base::Status setConfig(const std::string &key, const base::Value &value);
  base::Status setConfig(const std::string &key, base::Value &value);

  virtual base::Status init() = 0;
  virtual base::Status deinit() = 0;

  virtual base::Status preRun(base::ShapeMap min_shape = base::ShapeMap(),
                            base::ShapeMap opt_shape = base::ShapeMap(),
                            base::ShapeMap max_shape = base::ShapeMap()) = 0;
  virtual base::Status postRun() = 0;

  base::Status getStaticShape(base::ShapeMap shape_map);
  base::Status getMinShape(base::ShapeMap &shape_map);
  base::Status getOptShape(base::ShapeMap &shape_map);
  base::Status getCurentShape(base::ShapeMap &shape_map);
  base::Status getMaxShape(base::ShapeMap &shape_map);

  base::Status reShape(base::ShapeMap &shape_map);

  base::Status setDevice(device::Device *device);
  device::Device *getDevice();

  base::Status setMemoryPool(device::MemoryPool *memory_pool);
  device::MemoryPool *getMemoryPool();

  int64_t GetWorkspaceSize();
  base::Status setWorkspace(device::Buffer *buffer);

  int64_t getMemorySize();
  base::Status setMemory(device::Buffer *buffer);

  TensorMap getAllInputTensor();
  TensorMap getAllOutputTensor();

  int getNumOfInputTensor();
  int getNumOfOutputTensor();

  std::vector<std::string> getInputTensorNames();
  std::vector<std::string> getOutputTensorNames();

  std::shared_ptr<Tensor> getInputTensor(const std::string &key);
  std::shared_ptr<Tensor> getOutputTensor(const std::string &key);

  base::Status setInputTensor(const std::string &key,
                              const std::shared_ptr<Tensor> input_tensor);
  //
  std::shared_ptr<Tensor> getOutputTensor(const std::string &key,
                                          std::vector<int32_t> config);

  base::Status run();
  base::Status asyncRun();

  std::shared_ptr<ForwardImpl> getForwardImpl();

 private:
  std::shared_ptr<ForwardImpl> forward_impl_;
};

}  // namespace inference
}  // namespace nndeploy

#endif