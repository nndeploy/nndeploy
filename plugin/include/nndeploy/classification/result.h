
#ifndef _NNDEPLOY_CLASSIFICATION_RESULT_H_
#define _NNDEPLOY_CLASSIFICATION_RESULT_H_

#include "nndeploy/base/param.h"
#include "nndeploy/device/tensor.h"

namespace nndeploy {
namespace classification {

/**
 * @brief Classification Result
 *
 */
class NNDEPLOY_CC_API ClassificationLableResult : public base::Param {
 public:
  ClassificationLableResult(){};
  virtual ~ClassificationLableResult() {}

  int index_ = 0;
  int label_ids_ = -1;
  float scores_ = 0.0f;
  float feature_ = 0.0f;
};

class NNDEPLOY_CC_API ClassificationResult : public base::Param {
 public:
  ClassificationResult(){};
  virtual ~ClassificationResult(){};
  std::vector<ClassificationLableResult> labels_;
};

}  // namespace classification
}  // namespace nndeploy

#endif /* _NNDEPLOY_CLASSIFICATION_RESULT_H_ */
