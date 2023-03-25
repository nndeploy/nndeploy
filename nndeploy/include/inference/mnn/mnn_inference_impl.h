
#ifndef _NNDEPLOY_INCLUDE_INFERENCE_ABSTRACT_INFERENCE_IMPL_H_
#define _NNDEPLOY_INCLUDE_INFERENCE_ABSTRACT_INFERENCE_IMPL_H_

#include "nndeploy/include/base/basic.h"
#include "nndeploy/include/base/log.h"
#include "nndeploy/include/base/macro.h"
#include "nndeploy/include/base/object.h"
#include "nndeploy/include/base/status.h"
#include "nndeploy/include/base/value.h"
#include "nndeploy/include/device/device.h"
#include "nndeploy/include/device/tensor.h"
#include "nndeploy/include/inference/config.h"

namespace nndeploy {
namespace inference {

class AbstractInferenceImpl {
 public:
  AbstractInferenceImpl() = default;
  virtual ~AbstractInferenceImpl();

  virtual base::Status init(const Config &config) = 0;
  virtual base::Status deinit() = 0;

  virtual base::Status preRun(base::ShapeMap min_shape = base::ShapeMap(),
                              base::ShapeMap opt_shape = base::ShapeMap(),
                              base::ShapeMap max_shape = base::ShapeMap()) = 0;
  virtual base::Status postRun() = 0;

  virtual Config getConfig();

  virtual base::Status getStaticShape(base::ShapeMap &shape_map);
  virtual base::Status getMinShape(base::ShapeMap &shape_map);
  virtual base::Status getOptShape(base::ShapeMap &shape_map);
  virtual base::Status getCurentShape(base::ShapeMap &shape_map);
  virtual base::Status getMaxShape(base::ShapeMap &shape_map);

  virtual base::Status reShape(base::ShapeMap &shape_map) = 0;

  virtual base::Status setDevice(device::Device *device);
  virtual device::Device *getDevice();
  virtual device::Device *getDevice(int index);

  virtual base::Status setBufferPool(device::BufferPool *buffer_pool);
  virtual device::BufferPool *getBufferPool();
  virtual device::BufferPool *getBufferPool(int index);

  virtual int64_t getWorkspaceSize();
  virtual int64_t getWorkspaceSize(int index);
  virtual base::Status setWorkspace(device::Buffer *buffer);

  virtual int64_t getMemorySize();
  virtual int64_t getMemorySize(int index);
  virtual base::Status setMemory(device::Buffer *buffer);

  virtual device::TensorMap getAllInputTensor();
  virtual device::TensorMap getAllOutputTensor();

  virtual int getNumOfInputTensor();
  virtual int getNumOfOutputTensor();

  virtual std::vector<std::string> getInputTensorNames();
  virtual std::vector<std::string> getOutputTensorNames();

  virtual std::shared_ptr<device::Tensor> getInputTensor(
      const std::string &name);
  virtual std::shared_ptr<device::Tensor> getOutputTensor(
      const std::string &name);

  virtual base::Status setInputTensor(
      const std::string &name,
      const std::shared_ptr<device::Tensor> input_tensor) = 0;
  //
  virtual std::shared_ptr<device::Tensor> getOutputTensor(
      const std::string &name, std::vector<int32_t> config) = 0;

  virtual base::Status run() = 0;
  virtual base::Status asyncRun() = 0;

 protected:
  Config config_;

  base::ShapeMap static_shape_;
  base::ShapeMap current_shape_ = base::ShapeMap();
  base::ShapeMap min_shape_ = base::ShapeMap();
  base::ShapeMap opt_shape_ = base::ShapeMap();
  base::ShapeMap max_shape_ = base::ShapeMap();

  device::TensorMap current_input_tensors_;
  device::TensorMap current_output_tensors_;

  device::TensorMap max_input_tensors_;
  device::TensorMap max_output_tensors_;

  std::vector<device::Device *> device_;
  std::vector<device::BufferPool *> buffer_pool_;
};

class InferenceCreator {
 public:
  virtual ~InferenceCreator(){};
  virtual AbstractInferenceImpl *createInference() = 0;
};

template <typename T>
class TypeInferenceCreator : public InferenceCreator {
  virtual AbstractInferenceImpl *createInference() { return new T(); }
};

std::map<base::InferenceType, std::shared_ptr<InferenceCreator>>
    &getGlobalInferenceCreatorMap();

template <typename T>
class TypeInferenceRegister {
 public:
  explicit TypeInferenceRegister(base::InferenceType type) {
    getGlobalInferenceCreatorMap()[type] = std::shared_ptr<T>(new T());
  }
};

AbstractInferenceImpl *createInference(base::InferenceType type);

}  // namespace inference
}  // namespace nndeploy

#endif