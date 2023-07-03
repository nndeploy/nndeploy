
#ifndef _NNDEPLOY_INFERENCE_ABSTRACT_INFERENCE_H_
#define _NNDEPLOY_INFERENCE_ABSTRACT_INFERENCE_H_

#include "nndeploy/base/basic.h"
#include "nndeploy/base/glic_stl_include.h"
#include "nndeploy/base/log.h"
#include "nndeploy/base/macro.h"
#include "nndeploy/base/object.h"
#include "nndeploy/base/status.h"
#include "nndeploy/base/string.h"
#include "nndeploy/base/value.h"
#include "nndeploy/device/buffer.h"
#include "nndeploy/device/buffer_pool.h"
#include "nndeploy/device/device.h"
#include "nndeploy/device/tensor.h"
#include "nndeploy/inference/inference_param.h"

namespace nndeploy {
namespace inference {

class AbstractInference {
 public:
  AbstractInference(base::InferenceType type);
  virtual ~AbstractInference();

  base::InferenceType getInferenceType();

  base::Param *getParam();

  virtual base::Status init() = 0;
  virtual base::Status deinit() = 0;

  virtual bool isDynamicShape();
  base::ShapeMap getMinShape();
  base::ShapeMap getOptShape();
  base::ShapeMap getMaxShape();
  virtual base::Status reshape(base::ShapeMap &shape_map) = 0;

  virtual int64_t getMemorySize();
  virtual int64_t getMemorySize(int index);
  virtual base::Status setMemory(device::Buffer *buffer);

  virtual float getGFLOPs();
  virtual bool isBatch();
  virtual bool isShareCommanQueue();
  virtual bool isInputDynamic();
  virtual bool isOutputDynamic();
  virtual bool canOpInput();
  virtual bool canOpOutput();

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
                                      device::Tensor *input_tensor);
  virtual base::Status setOutputTensor(const std::string &name,
                                       device::Tensor *output_tensor);

  virtual base::Status run() = 0;

 protected:
  base::InferenceType type_;
  InferenceParam *inference_param_;

  bool is_share_command_queue_ = false;
  bool is_batch_ = false;
  bool is_input_dynamic_ = false;
  bool is_output_dynamic_ = false;
  bool can_op_input_ = false;
  bool can_op_output_ = false;

  std::map<std::string, device::Tensor *> input_tensors_;
  std::map<std::string, device::Tensor *> output_tensors_;

  std::map<std::string, device::Tensor *> external_input_tensors_;
  std::map<std::string, device::Tensor *> external_output_tensors_;
};

class InferenceCreator {
 public:
  virtual ~InferenceCreator(){};
  virtual AbstractInference *createInference(base::InferenceType type) = 0;
};

template <typename T>
class TypeInferenceCreator : public InferenceCreator {
  virtual AbstractInference *createInference(base::InferenceType type) {
    return new T(type);
  }
};

std::map<base::InferenceType, std::shared_ptr<InferenceCreator>> &
getGlobalInferenceCreatorMap();

template <typename T>
class TypeInferenceRegister {
 public:
  explicit TypeInferenceRegister(base::InferenceType type) {
    getGlobalInferenceCreatorMap()[type] = std::shared_ptr<T>(new T());
  }
};

AbstractInference *createInference(base::InferenceType type);

}  // namespace inference
}  // namespace nndeploy

#endif