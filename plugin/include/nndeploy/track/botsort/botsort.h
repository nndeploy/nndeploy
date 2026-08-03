
#ifndef _NNDEPLOY_TRACK_BOTSORT_BOTSORT_H_
#define _NNDEPLOY_TRACK_BOTSORT_BOTSORT_H_

#include <opencv2/core/core.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/video/tracking.hpp>
#include <vector>

#include "nndeploy/track/bytetrack/bytetrack.h"

namespace nndeploy {
namespace track {

/**
 * @brief Bot-SORT: Robust Multi-Object Tracking with Camera Motion
 * Compensation
 *
 * Reference: https://github.com/NirAharon/BoT-SORT
 * Paper: BoT-SORT: Robust Associations Multi-Pedestrian Tracking
 *
 * Extends ByteTrack with:
 *   1. Global Motion Compensation (GMC) using feature matching
 *   2. Camera motion- compensated Kalman filter prediction
 */
class NNDEPLOY_CC_API BotSORT : public ByteTrack {
 public:
  BotSORT();
  virtual ~BotSORT() {}

  virtual MOTResult update(const std::vector<cv::Vec4f> &ltrb_boxes,
                           const std::vector<float> &scores,
                           const std::vector<int> &class_ids) override;

  virtual void setFrame(const cv::Mat &img);

 private:
  cv::Mat compute_gmc(const cv::Mat &curr_img);
  void apply_gmc_to_track(STrack &track, const cv::Mat &affine);

  virtual void multi_predict() override;

  cv::Mat curr_frame_;
  cv::Mat prev_gray_;
  cv::Ptr<cv::ORB> orb_;
  float gm_scale_;
};

}  // namespace track
}  // namespace nndeploy

#endif  // _NNDEPLOY_TRACK_BOTSORT_BOTSORT_H_
