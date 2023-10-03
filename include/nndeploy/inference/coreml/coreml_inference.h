
#ifndef _NNDEPLOY_INFERENCE_COREML_COREML_INFERENCE_H_
#define _NNDEPLOY_INFERENCE_COREML_COREML_INFERENCE_H_

#include "nndeploy/base/common.h"
#include "nndeploy/base/log.h"
#include "nndeploy/base/macro.h"
#include "nndeploy/base/object.h"
#include "nndeploy/base/shape.h"
#include "nndeploy/base/status.h"
#include "nndeploy/base/value.h"
#include "nndeploy/device/device.h"
#include "nndeploy/device/tensor.h"
#include "nndeploy/inference/coreml/coreml_convert.h"
#include "nndeploy/inference/coreml/coreml_include.h"
#include "nndeploy/inference/coreml/coreml_inference_param.h"
#include "nndeploy/inference/inference.h"
#include "nndeploy/inference/inference_param.h"

namespace nndeploy {
namespace inference {

#define CHECK_ERR(err) \
  if (err) NSLog(@"error: %@", err);

class CoremlInference : public Inference {
 public:
  CoremlInference(base::InferenceType type);
  virtual ~CoremlInference();

  virtual base::Status init();
  virtual base::Status deinit();

  virtual base::Status reshape(base::ShapeMap &shape_map);

  virtual int64_t getMemorySize();

  virtual float getGFLOPs();

  virtual device::TensorDesc getInputTensorAlignDesc(const std::string &name);
  virtual device::TensorDesc getOutputTensorAlignDesc(const std::string &name);

  virtual base::Status run();

 private:
  base::Status allocateInputOutputTensor();
  base::Status deallocateInputOutputTensor();
  MLModel *mlmodel_ = nullptr;
  NSError *err_ = nil;
  MLModelConfiguration *config_ = nullptr;
  NSMutableDictionary *dict_ = nullptr;
  NSMutableDictionary *result_ = nullptr;
};

}  // namespace inference
}  // namespace nndeploy

#endif
