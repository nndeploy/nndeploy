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
#ifndef _NNDEPLOY_INCLUDE_INFERENCE_FORWARD_IMPL_H_
#define _NNDEPLOY_INCLUDE_INFERENCE_FORWARD_IMPL_H_

#include "nndeploy/include/base/log.h"
#include "nndeploy/include/base/macro.h"
#include "nndeploy/include/base/object.h"
#include "nndeploy/include/base/status.h"
#include "nndeploy/include/base/type.h"
#include "nndeploy/include/base/value.h"
#include "nndeploy/include/architecture/device.h"
#include "nndeploy/include/inference/config.h"
#include "nndeploy/include/inference/tensor.h"

namespace nndeploy {
namespace inference {

class ForwardImpl {
 public:
  ForwardImpl(Config config);

  virtual ~ForwardImpl();

  base::Status setConfig(const std::string &key, const base::Value &value);
  base::Status setConfig(const std::string &key, base::Value &value);

  bool isConstract();

  virtual base::Status init(base::ShapeMap min_shape = base::ShapeMap(),
                            base::ShapeMap opt_shape = base::ShapeMap(),
                            base::ShapeMap max_shape = base::ShapeMap()) = 0;
  virtual base::Status deinit() = 0;

  virtual base::Status getStaticShape(base::ShapeMap shape_map);
  virtual base::Status getMinShape(base::ShapeMap &shape_map);
  virtual base::Status getOptShape(base::ShapeMap &shape_map);
  virtual base::Status getCurentShape(base::ShapeMap &shape_map);
  virtual base::Status getMaxShape(base::ShapeMap &shape_map);

  virtual base::Status reShape(base::ShapeMap &shape_map);

  virtual base::Status setDevice(architecture::Device *device);
  virtual architecture::Device *getDevice();

  virtual base::Status setMemoryPool(architecture::MemoryPool *memory_pool);
  virtual architecture::MemoryPool *getMemoryPool();

  virtual int64_t GetWorkspaceSize();
  virtual base::Status setWorkspace(architecture::Buffer *buffer);

  virtual int64_t getMemorySize();
  virtual base::Status setMemory(architecture::Buffer *buffer);

  virtual TensorMap getAllInputTensor();
  virtual TensorMap getAllOutputTensor();

  virtual int getNumOfInputTensor();
  virtual int getNumOfOutputTensor();

  virtual std::vector<std::string> getInputTensorNames();
  virtual std::vector<std::string> getOutputTensorNames();

  virtual std::shared_ptr<Tensor> getInputTensor(const std::string &key);
  virtual std::shared_ptr<Tensor> getOutputTensor(const std::string &key);

  virtual base::Status setInputTensor(
      const std::string &key, const std::shared_ptr<Tensor> input_tensor);
  //
  virtual std::shared_ptr<Tensor> getOutputTensor(const std::string &key,
                                                  std::vector<int32_t> config);

  virtual base::Status run();
  virtual base::Status asyncRun();

 protected:
  Config config_;
  base::ShapeMap static_shape_map_;
  bool is_construct_flag_ = false;

  base::ShapeMap current_shape_ = base::ShapeMap();
  base::ShapeMap min_shape_ = base::ShapeMap();
  base::ShapeMap opt_shape_ = base::ShapeMap();
  base::ShapeMap max_shape_ = base::ShapeMap();

  TensorMap current_input_tensors;
  TensorMap current_output_tensors;

  TensorMap max_input_tensors;
  TensorMap max_output_tensors;

  architecture::Device *device_ = nullptr;
  architecture::MemoryPool *memory_pool = nullptr;
};

}  // namespace inference
}  // namespace nndeploy

#endif