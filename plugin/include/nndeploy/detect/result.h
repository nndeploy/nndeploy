
#ifndef _NNDEPLOY_DETECT_DETECT_RESULT_H_
#define _NNDEPLOY_DETECT_DETECT_RESULT_H_

namespace nndeploy {
namespace detect {

/**
 * @brief Detect Result
 *
 */
class NNDEPLOY_CC_API DetectBBoxResult : public base::Param {
 public:
  int index_;
  int label_id_;
  float score_;
  std::array<float, 4> bbox_;  // xmin, ymin, xmax, ymax
  device::Tensor mask_;
};

class NNDEPLOY_CC_API DetectResult : public base::Param {
 public:
  std::vector<DetectBBoxResult> bboxs_;
};

}  // namespace detect
}  // namespace nndeploy

#endif /* _NNDEPLOY_DETECT_DETECT_RESULT_H_ */
