
#ifndef _NNDEPLOY_PIPELINE_INFER_INFER_H_
#define _NNDEPLOY_PIPELINE_INFER_INFER_H_

#include "nndeploy/inference/inference.h"
#include "nndeploy/inference/inference_param.h"
#include "nndeploy/pipeline/task.h"

namespace nndeploy {
namespace pipeline {

class NNDEPLOY_CC_API Infer : public Task {
 public:
  Infer(const std::string &name, base::InferenceType type, Packet *input,
        Packet *output);
  virtual ~Infer();

  virtual base::Status setParam(base::Param *param);
  virtual base::Param *getParam();

  virtual base::Status init();
  virtual base::Status deinit();
  virtual base::Status reshape();
  virtual base::Status run();

 private:
  template <bool is_input_dynamic, bool is_output_dynamic, bool can_op_input,
            bool can_op_output>
  base::Status initTemplate();
  template <bool is_input_dynamic, bool is_output_dynamic, bool can_op_input,
            bool can_op_output>
  base::Status deinitTemplate();
  template <bool is_input_dynamic, bool is_output_dynamic, bool can_op_input,
            bool can_op_output>
  base::Status reshapeTemplate();

  base::Status initDefault();
  base::Status deinitDefault();
  base::Status reshapeDefault();
  base::Status runDefault();

 private:
  base::InferenceType type_;
  inference::Inference *inference_ = nullptr;

  std::vector<device::Tensor *> input_tensors_;
  std::vector<device::Tensor *> output_tensors_;

  bool is_input_dynamic_ = false;
  bool is_output_dynamic_ = false;
  bool can_op_input_ = false;
  bool can_op_output_ = false;
};

}  // namespace pipeline
}  // namespace nndeploy

#endif /* _NNDEPLOY_PIPELINE_INFER_INFER_H_ */
