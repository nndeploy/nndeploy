
#ifndef _NNDEPLOY_INFERENCE_MDC_MDC_INFERENCE_H_
#define _NNDEPLOY_INFERENCE_MDC_MDC_INFERENCE_H_

#include "nndeploy/base/common.h"
#include "nndeploy/base/log.h"
#include "nndeploy/base/macro.h"
#include "nndeploy/base/object.h"
#include "nndeploy/base/status.h"
#include "nndeploy/base/value.h"
#include "nndeploy/device/device.h"
#include "nndeploy/device/tensor.h"
#include "nndeploy/inference/inference.h"
#include "nndeploy/inference/inference_param.h"
#include "nndeploy/inference/mdc/mdc_include.h"
#include "nndeploy/inference/mdc/mdc_inference_param.h"

namespace nndeploy {
namespace inference {

struct OrtValueInfo {
  std::string name;
  std::vector<int64_t> shape;
  aclDataType dtype;
};

class MdcInference : public Inference {
 public:
  MdcInference(base::InferenceType type);
  virtual ~MdcInference();

  virtual base::Status init();
  virtual base::Status deinit();

  virtual base::Status reshape(base::ShapeMap &shape_map);

  virtual base::Status run();

  virtual device::Tensor *getOutputTensorAfterRun(const std::string &name, base::DeviceType device_type, bool is_copy,
                                                  base::DataFormat data_format = base::kDataFormatAuto);

 private:
  bool isDynamic(std::vector<int64_t> &shape);
  virtual void ReleaseAllResource();

 private:
  int batch_size_ = 1;

  const char *aclConfigPath = "";  // json文件，如果要使用msprof工具分析模型各算子执行时间时需要指定，格式看mdc文档

  aclrtContext context_ = nullptr;

  aclmdlDesc *modelDesc_ = nullptr;
  aclmdlDataset *inputDataset_ = nullptr;
  aclmdlDataset *outputDataset_ = nullptr;

  uint32_t modelId_;

  std::vector<OrtValueInfo> inputs_desc_;
  std::vector<OrtValueInfo> outputs_desc_;

  std::map<std::string, device::Tensor *> max_input_tensors_;
  std::map<std::string, device::Tensor *> max_output_tensors_;
  std::map<std::string, std::string> mdc_change_output_names_;
};

}  // namespace inference
}  // namespace nndeploy

#endif
