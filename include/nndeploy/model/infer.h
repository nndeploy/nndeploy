
#ifndef _NNDEPLOY_MODEL_INFER_H_
#define _NNDEPLOY_MODEL_INFER_H_

#include "nndeploy/dag/task.h"
#include "nndeploy/inference/inference.h"
#include "nndeploy/inference/inference_param.h"

namespace nndeploy {
namespace model {

class NNDEPLOY_CC_API Infer : public dag::Task {
 public:
  Infer(const std::string &name, base::InferenceType type, dag::Packet *input,
        dag::Packet *output);
  Infer(const std::string &name, base::InferenceType type,
        std::vector<dag::Packet *> inputs, std::vector<dag::Packet *> outputs);
  virtual ~Infer();

  virtual base::Status setParam(base::Param *param);
  virtual base::Param *getParam();

  virtual base::Status init();
  virtual base::Status deinit();
  virtual base::Status reshape();
  virtual base::Status run();

  virtual inference::Inference *getInference();

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

}  // namespace model
}  // namespace nndeploy

#endif /* _NNDEPLOY_MODEL_INFER_H_ */
