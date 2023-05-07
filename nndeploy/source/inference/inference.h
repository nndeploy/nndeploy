
#ifndef _NNDEPLOY_SOURCE_INFERENCE_INFERENCE_H_
#define _NNDEPLOY_SOURCE_INFERENCE_INFERENCE_H_

#include "nndeploy/source/base/basic.h"
#include "nndeploy/source/base/glic_stl_include.h"
#include "nndeploy/source/base/log.h"
#include "nndeploy/source/base/macro.h"
#include "nndeploy/source/base/object.h"
#include "nndeploy/source/base/status.h"
#include "nndeploy/source/base/string.h"
#include "nndeploy/source/base/value.h"
#include "nndeploy/source/device/architecture.h"
#include "nndeploy/source/device/buffer.h"
#include "nndeploy/source/device/buffer_pool.h"
#include "nndeploy/source/device/device.h"
#include "nndeploy/source/device/tensor.h"
#include "nndeploy/source/inference/inference_param.h"

namespace nndeploy {
namespace inference {

class Inference {
 public:
  Inference(base::InferenceType type);
  virtual ~Inference();

  base::InferenceType getInferenceType();

  InferenceParam *getInferenceParam();

  virtual base::Status init() = 0;
  virtual base::Status deinit() = 0;

  virtual base::Status getMinShape(base::ShapeMap &shape_map);
  virtual base::Status getOptShape(base::ShapeMap &shape_map);
  virtual base::Status getCurentShape(base::ShapeMap &shape_map);
  virtual base::Status getMaxShape(base::ShapeMap &shape_map);

  virtual base::Status reShape(base::ShapeMap &shape_map) = 0;

  virtual int64_t getMemorySize();
  virtual int64_t getMemorySize(int index);
  virtual base::Status setMemory(device::Buffer *buffer);

  virtual float getGFLOPs();

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

 protected:
  base::InferenceType type_;
  InferenceParam *inference_param_;

  base::ShapeMap current_shape_ = base::ShapeMap();
  base::ShapeMap min_shape_ = base::ShapeMap();
  base::ShapeMap opt_shape_ = base::ShapeMap();
  base::ShapeMap max_shape_ = base::ShapeMap();

  device::TensorMap current_input_tensors_;
  device::TensorMap current_output_tensors_;

  device::TensorMap max_input_tensors_;
  device::TensorMap max_output_tensors_;
};

class InferenceCreator {
 public:
  virtual ~InferenceCreator(){};
  virtual Inference *createInference(base::InferenceType type) = 0;
};

template <typename T>
class TypeInferenceCreator : public InferenceCreator {
  virtual Inference *createInference(base::InferenceType type) {
    return new T(type);
  }
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

Inference *createInference(base::InferenceType type);

}  // namespace inference
}  // namespace nndeploy

#endif