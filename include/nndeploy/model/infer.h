
#ifndef _NNDEPLOY_MODEL_INFER_H_
#define _NNDEPLOY_MODEL_INFER_H_

#include "nndeploy/dag/node.h"
#include "nndeploy/inference/inference.h"
#include "nndeploy/inference/inference_param.h"

namespace nndeploy {
namespace model {

class NNDEPLOY_CC_API Infer : public dag::Node {
 public:
  Infer(const std::string &name, base::InferenceType type, dag::Edge *input,
        dag::Edge *output);
  Infer(const std::string &name, base::InferenceType type,
        std::initializer_list<dag::Edge *> inputs,
        std::initializer_list<dag::Edge *> outputs);
  Infer(const std::string &name, base::InferenceType type,
        std::vector<dag::Edge *> inputs, std::vector<dag::Edge *> outputs);
  virtual ~Infer();

  virtual base::Status setParam(base::Param *param);
  virtual base::Param *getParam();

  virtual base::Status init();
  virtual base::Status deinit();

  virtual int64_t getMemorySize();
  virtual base::Status setMemory(device::Buffer *buffer);

  virtual base::Status run();

  virtual inference::Inference *getInference();

 private:
  base::InferenceType type_;
  inference::Inference *inference_ = nullptr;

  bool is_input_dynamic_ = false;
  bool is_output_dynamic_ = false;
  bool can_op_input_ = false;
  bool can_op_output_ = false;
};

}  // namespace model
}  // namespace nndeploy

#endif /* _NNDEPLOY_MODEL_INFER_H_ */
