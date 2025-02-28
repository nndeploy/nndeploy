
#ifndef _NNDEPLOY_INFER_INFER_H_
#define _NNDEPLOY_INFER_INFER_H_

#include "nndeploy/dag/node.h"
#include "nndeploy/inference/inference.h"
#include "nndeploy/inference/inference_param.h"

namespace nndeploy {
namespace infer {

class NNDEPLOY_CC_API Infer : public dag::Node {
 public:
  Infer(const std::string &name, base::InferenceType type, dag::Edge *input,
        dag::Edge *output);
  Infer(const std::string &name, base::InferenceType type,
        std::initializer_list<dag::Edge *> inputs,
        std::initializer_list<dag::Edge *> outputs);
  Infer(const std::string &name, base::InferenceType type,
        std::vector<dag::Edge *> inputs, std::vector<dag::Edge *> outputs);

  Infer(const std::string &name);
  Infer(const std::string &name, std::initializer_list<dag::Edge *> inputs,
        std::initializer_list<dag::Edge *> outputs);
  Infer(const std::string &name, std::vector<dag::Edge *> inputs,
        std::vector<dag::Edge *> outputs);

  virtual ~Infer();

  virtual base::Status setInferenceType(base::InferenceType inference_type);

  virtual base::Status setParam(base::Param *param);
  virtual base::Param *getParam();

  virtual base::Status setParamSharedPtr(std::shared_ptr<base::Param> param);
  virtual std::shared_ptr<base::Param> getParamSharedPtr();

  virtual base::Status init();
  virtual base::Status deinit();

  virtual int64_t getMemorySize();
  virtual base::Status setMemory(device::Buffer *buffer);

  virtual base::Status run();

  virtual std::shared_ptr<inference::Inference> getInference();

 private:
  base::InferenceType type_;
  std::shared_ptr<inference::Inference> inference_ = nullptr;

  bool is_input_dynamic_ = false;
  bool is_output_dynamic_ = false;
  bool can_op_input_ = false;
  bool can_op_output_ = false;
};

}  // namespace infer
}  // namespace nndeploy

#endif /* _NNDEPLOY_INFER_INFER_H_ */
