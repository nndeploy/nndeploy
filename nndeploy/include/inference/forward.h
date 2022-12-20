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
#ifndef _NNDEPLOY_INCLUDE_INFERENCE_ABSTRACT_MODEL_
#define _NNDEPLOY_INCLUDE_INFERENCE_ABSTRACT_MODEL_

#include "nndeploy/include/base/config.h"
#include "nndeploy/include/base/status.h"
#include "nndeploy/include/inference/abstract_session.h"


using namespace nndeploy::base;
using namespace nndeploy::device;

namespace nndeploy {
namespace inference {

class Forward {
 public:
  explicit Forward(InferenceConfig inference_config);

  virtual ~Forward();

  InferenceConfig GetConfig();

  bool isConstract();

  virtual Status Init(ShapeMap min_shape = ShapeMap(), ShapeMap opt_shape = ShapeMap(),
      ShapeMap max_shape = ShapeMap()) = 0;
  virtual Status Deinit() = 0;

  virtual Status GetStaticShape(ShapeMap shape_map);
  virtual Status GetMinShape(ShapeMap &shape_map);
  virtual Status GetOptShape(ShapeMap &shape_map);
  virtual Status GetCurentShape(ShapeMap &shape_map);
  virtual Status GetMaxShape(ShapeMap &shape_map);

  virtual Status ReShape(ShapeMap &shape_map);

  virtual Stauts SetDevice(DevicePacket *device);
  virtual Device *GetDevice();

  virtual Stauts SetWorkspace(Buffer *buffer);
  virtual int64_t GetWorkspaceSize();

  virtual Stauts SetMemory(Buffer *buffer);
  virtual int64_t GetMemorySize();

  // 得到内部分配的内存
  virtual Status GetAllInputTensor(TensorMap input_tensors);
  virtual Status GetAllOutputTensor(TensorMap output_tensors);

  virtual int GetNumOfInputTensor();
  virtual int GetNumOfOutputTensor();

  virtual std::vector<std::string> GetInputTensorNames();
  virtual std::vector<std::string> GetOutputTensorNames();

  virtual std::shared_ptr<Tensor> GetInputTensor(const string &key);
  virtual std::shared_ptr<Tensor> GetOutputTensor(const string &key);

  // 外部分配内存
  virtual Status SetInputTensor(const string &key,
                                const std::shared_ptr<Tensor> input_tensor);
  //
  virtual std::shared_ptr<Tensor> GetOutputTensor(const string &key,
                                                  std::vector<int32_t> config);

  virtual Status Run();
  virtual Stauts AsyncRun();

 private:
  std::shared_ptr<ForwardImpl> forward_impl_ = nullptr;  // 通常cpu runtime 不用导入
};

}  // namespace inference
}  // namespace nndeploy

#endif