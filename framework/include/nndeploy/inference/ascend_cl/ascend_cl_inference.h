
#ifndef _NNDEPLOY_INFERENCE_ASCEND_CL_INFERENCE_H_
#define _NNDEPLOY_INFERENCE_ASCEND_CL_INFERENCE_H_

#include "nndeploy/base/any.h"
#include "nndeploy/base/common.h"
#include "nndeploy/base/log.h"
#include "nndeploy/base/macro.h"
#include "nndeploy/base/object.h"
#include "nndeploy/base/status.h"
#include "nndeploy/device/device.h"
#include "nndeploy/device/tensor.h"
#include "nndeploy/inference/ascend_cl/ascend_cl_include.h"
#include "nndeploy/inference/ascend_cl/ascend_cl_inference_param.h"
#include "nndeploy/inference/inference.h"
#include "nndeploy/inference/inference_param.h"

namespace nndeploy {
namespace inference {

class AscendCLInference : public Inference {
 public:
  AscendCLInference(base::InferenceType type);
  virtual ~AscendCLInference();

  virtual base::Status init();
  virtual base::Status deinit();

  virtual base::Status reshape(base::ShapeMap &shape_map);

  virtual base::Status run();

  virtual device::Tensor *getOutputTensorAfterRun(
      const std::string &name, base::DeviceType device_type, bool is_copy,
      base::DataFormat data_format = base::kDataFormatAuto);

 private:
  bool isDynamic(std::vector<int64_t> &shape);
  virtual void releaseAllResource();

 private:
  int batch_size_ = 1;

  aclrtContext context_ = nullptr;

  aclmdlDesc *model_desc_ = nullptr;
  aclmdlDataset *input_dataset_ = nullptr;
  aclmdlDataset *output_dataset_ = nullptr;

  uint32_t model_id_;

  std::map<std::string, device::Tensor *> max_input_tensors_;
  std::map<std::string, device::Tensor *> max_output_tensors_;
  std::map<std::string, std::string> ascend_cl_change_output_names_;
};

}  // namespace inference
}  // namespace nndeploy

#endif
