
#ifndef _NNDEPLOY_TASK_INFERENCE_INFERENCE_H_
#define _NNDEPLOY_TASK_INFERENCE_INFERENCE_H_

#include "nndeploy/inference/abstract_inference.h"
#include "nndeploy/inference/inference_param.h"
#include "nndeploy/task/task.h"

namespace nndeploy {
namespace task {

class Inference : public Task {
 public:
  Inference(const std::string &name, base::InferenceType type, Packet *input,
            Packet *output);
  virtual ~Inference();

  virtual base::Status setParam(base::Param *param);
  virtual base::Param *getParam();

  virtual base::Status init();
  virtual base::Status deinit();
  virtual base::Status reShape();
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
  base::Status reShapeTemplate();

  base::Status initDefault();
  base::Status deinitDefault();
  base::Status reShapeDefault();
  base::Status runDefault();

 private:
  base::InferenceType type_;
  inference::AbstractInference *abstract_inference_ = nullptr;

  std::vector<device::Tensor *> input_tensors_;
  std::vector<device::Tensor *> output_tensors_;

  bool is_input_dynamic_ = false;
  bool is_output_dynamic_ = false;
  bool can_op_input_ = false;
  bool can_op_output_ = false;
};

}  // namespace task
}  // namespace nndeploy

#endif
