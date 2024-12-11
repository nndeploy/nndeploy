
#ifndef _NNDEPLOY_CLASSIFICATION_RESULT_H_
#define _NNDEPLOY_CLASSIFICATION_RESULT_H_

#include "nndeploy/base/param.h"
#include "nndeploy/device/tensor.h"

namespace nndeploy {
namespace llm {

/**
 * @brief LLM Result
 *
 */
class NNDEPLOY_CC_API LlmTokenResult : public base::Param {
 public:
  LlmTokenResult(){};
  virtual ~LlmTokenResult() {}

  int index_ = 0;
  int label_ids_ = -1;
  float scores_ = 0.0f;
  float feature_ = 0.0f;
};

class NNDEPLOY_CC_API LlmResult : public base::Param {
 public:
  LlmResult(){};
  virtual ~LlmResult(){};
  std::vector<LlmTokenResult> labels_;
};

}  // namespace llm  
}  // namespace nndeploy

#endif /* _NNDEPLOY_CLASSIFICATION_RESULT_H_ */
