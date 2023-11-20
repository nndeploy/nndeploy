
#ifndef _NNDEPLOY_INFERENCE_NCNN_NCNN_INFERENCE_H_
#define _NNDEPLOY_INFERENCE_NCNN_NCNN_INFERENCE_H_

#include "nndeploy/base/common.h"
#include "nndeploy/base/log.h"
#include "nndeploy/base/macro.h"
#include "nndeploy/base/object.h"
#include "nndeploy/base/shape.h"
#include "nndeploy/base/status.h"
#include "nndeploy/base/value.h"
#include "nndeploy/device/device.h"
#include "nndeploy/device/tensor.h"
#include "nndeploy/inference/inference.h"
#include "nndeploy/inference/inference_param.h"
#include "nndeploy/inference/ncnn/ncnn_convert.h"
#include "nndeploy/inference/ncnn/ncnn_include.h"
#include "nndeploy/inference/ncnn/ncnn_inference_param.h"

namespace nndeploy {
namespace inference {

class NcnnInference : public Inference {
 public:
  NcnnInference(base::InferenceType type);
  virtual ~NcnnInference();

  virtual base::Status init();
  virtual base::Status deinit();

  virtual base::Status reshape(base::ShapeMap &shape_map);

  virtual base::Status run();

  virtual device::Tensor *getOutputTensorAfterRun(const std::string &name,
                                                  bool is_copy);

 private:
  base::Status allocateInputOutputTensor();
  base::Status deallocateInputOutputTensor();

 private:
  ncnn::Net net_;
};

}  // namespace inference
}  // namespace nndeploy

#endif
