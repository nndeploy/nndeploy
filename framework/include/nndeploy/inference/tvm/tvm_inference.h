#ifndef _NNDEPLOY_INFERENCE_TVM_TVM_INFERENCE_H_
#define _NNDEPLOY_INFERENCE_TVM_TVM_INFERENCE_H_

#include "nndeploy/base/common.h"
#include "nndeploy/base/log.h"
#include "nndeploy/base/macro.h"
#include "nndeploy/base/object.h"
#include "nndeploy/base/status.h"
#include "nndeploy/base/value.h"
#include "nndeploy/base/shape.h"
#include "nndeploy/device/device.h"
#include "nndeploy/device/tensor.h"
#include "nndeploy/inference/inference.h"
#include "nndeploy/inference/inference_param.h"
#include "nndeploy/inference/tvm/tvm_include.h"
#include "nndeploy/inference/tvm/tvm_inference_param.h"

namespace nndeploy {
namespace inference {

/*!
 * \brief various meta information related to the compiled TVM model.
 */
typedef struct _TVMMetaInfo {
  int n_inputs;
  int n_outputs;
  std::map<std::string, std::pair<std::vector<int64_t>, std::string>>
      input_info;
  std::map<std::string, std::pair<std::vector<int64_t>, std::string>>
      output_info;
} TVMMetaInfo;

class TvmInference : public Inference {
 public:
  TvmInference(base::InferenceType type);
  virtual ~TvmInference();

  virtual base::Status init();
  virtual base::Status deinit();

  virtual base::Status reshape(base::ShapeMap &shape_map);

  virtual base::Status run();

  virtual device::Tensor *getOutputTensorAfterRun(
      const std::string &name, base::DeviceType device_type, bool is_copy,
      base::DataFormat data_format = base::kDataFormatAuto);

  size_t getMemSize(const tvm::runtime::NDArray &data);

 private:
  base::Status allocateInputOutputTensor();
  base::Status deallocateInputOutputTensor();

 private:
  tvm::runtime::Module mod_handle_;    // Module handle for the shared object
  tvm::runtime::Module graph_handle_;  // Graph runtime module handle

  TVMMetaInfo mInfo_;
};

}  // namespace inference
}  // namespace nndeploy
#endif