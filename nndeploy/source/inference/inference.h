
#ifndef _NNDEPLOY_SOURCE_INFERENCE_Inference_H_
#define _NNDEPLOY_SOURCE_INFERENCE_Inference_H_

#include "nndeploy/source/base/basic.h"
#include "nndeploy/source/base/log.h"
#include "nndeploy/source/base/macro.h"
#include "nndeploy/source/base/object.h"
#include "nndeploy/source/base/status.h"
#include "nndeploy/source/device/device.h"
#include "nndeploy/source/device/tensor.h"
#include "nndeploy/source/inference/abstract_inference_impl.h"
#include "nndeploy/source/inference/config.h"

namespace nndeploy {
namespace inference {

class Inference {
 public:
  explicit Inference(base::InferenceType type);
  ~Inference();

  base::Status init(const Config &config);
  base::Status deinit();

  base::Status preRun(base::ShapeMap min_shape = base::ShapeMap(),
                      base::ShapeMap opt_shape = base::ShapeMap(),
                      base::ShapeMap max_shape = base::ShapeMap());
  base::Status postRun();

  Config getConfig();

  base::Status getStaticShape(base::ShapeMap &shape_map);
  base::Status getMinShape(base::ShapeMap &shape_map);
  base::Status getOptShape(base::ShapeMap &shape_map);
  base::Status getCurentShape(base::ShapeMap &shape_map);
  base::Status getMaxShape(base::ShapeMap &shape_map);

  base::Status reShape(base::ShapeMap &shape_map);

  base::Status setDevice(device::Device *device);
  device::Device *getDevice();
  device::Device *getDevice(int index);

  base::Status setBufferPool(device::BufferPool *buffer_pool);
  device::BufferPool *getBufferPool();
  device::BufferPool *getBufferPool(int index);

  int64_t getWorkspaceSize();
  int64_t getWorkspaceSize(int index);
  base::Status setWorkspace(device::Buffer *buffer);

  int64_t getMemorySize();
  int64_t getMemorySize(int index);
  base::Status setMemory(device::Buffer *buffer);

  device::TensorMap getAllInputTensor();
  device::TensorMap getAllOutputTensor();

  int getNumOfInputTensor();
  int getNumOfOutputTensor();

  std::vector<std::string> getInputTensorNames();
  std::vector<std::string> getOutputTensorNames();

  std::shared_ptr<device::Tensor> getInputTensor(const std::string &name);
  std::shared_ptr<device::Tensor> getOutputTensor(const std::string &name);

  base::Status setInputTensor(
      const std::string &name,
      const std::shared_ptr<device::Tensor> input_tensor);
  //
  std::shared_ptr<device::Tensor> getOutputTensor(const std::string &name,
                                                  std::vector<int32_t> config);

  base::Status run();
  base::Status asyncRun();

  base::InferenceType getInferenceType();
  AbstractInferenceImpl *getInferenceImpl();

 private:
  base::InferenceType type_;
  AbstractInferenceImpl *inference_impl_;
};

}  // namespace inference
}  // namespace nndeploy

#endif