
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

  base::Param *getParam();

  virtual base::Status init() = 0;
  virtual base::Status deinit() = 0;

  virtual base::Status reshape(base::ShapeMap &shape_map) = 0;

  virtual int64_t getMemorySize();
  virtual int64_t getMemorySize(int index);
  virtual base::Status setMemory(device::Buffer *buffer);

  virtual float getGFLOPs();
  virtual bool isShareCommanQueue();

  virtual int getNumOfInputTensor();
  virtual int getNumOfOutputTensor();

  virtual std::string getInputName(int i);
  virtual std::string getOutputName(int i);
  virtual std::vector<std::string> getAllInputTensorName();
  virtual std::vector<std::string> getAllOutputTensorName();

  virtual base::IntVector getInputShape(const std::string &name);
  virtual base::ShapeMap getAllInputShape();

  virtual device::TensorDesc getInputTensorDesc(const std::string &name);
  virtual device::TensorDesc getOutputTensorDesc(const std::string &name);

  virtual device::TensorDesc getInputTensorAlignDesc(const std::string &name);
  virtual device::TensorDesc getOutputTensorAlignDesc(const std::string &name);

  virtual std::map<std::string, device::Tensor *> getAllInputTensorMap();
  virtual std::map<std::string, device::Tensor *> getAllOutputTensorMap();

  virtual std::vector<device::Tensor *> getAllInputTensorVector();
  virtual std::vector<device::Tensor *> getAllOutputTensorVector();

  virtual device::Tensor *getInputTensor(const std::string &name);
  virtual device::Tensor *getOutputTensor(const std::string &name);

  virtual base::Status setInputTensor(const std::string &name,
                                      device::Tensor *input_tensor) = 0;
  virtual base::Status setOutputTensor(const std::string &name,
                                       device::Tensor *output_tensor) = 0;

  virtual base::Status run() = 0;

 protected:
  base::InferenceType type_;
  InferenceParam *inference_param_;
  bool is_share_command_queue_ = false;

  std::map<std::string, device::Tensor *> input_tensors_;
  std::map<std::string, device::Tensor *> output_tensors_;
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