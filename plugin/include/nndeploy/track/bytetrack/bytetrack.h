
#ifndef _NNDEPLOY_TRACK_BYTETRACK_BYTETRACK_H_
#define _NNDEPLOY_TRACK_BYTETRACK_BYTETRACK_H_

#include <vector>

#include "nndeploy/track/result.h"
#include "nndeploy/track/trajectory.h"

namespace nndeploy {
namespace track {

/**
 * @brief A tracked state with embedded Kalman filter
 *
 * Kalman state: [cx, cy, w, h, vx, vy, vw, vh] (8-dim)
 * Measurement: [cx, cy, w, h] (4-dim)
 * Constant velocity model
 */
struct STrack {
  cv::Mat mean_;        // 8x1 state mean
  cv::Mat covariance_;  // 8x8 state covariance
  cv::Vec4f ltrb_;      // [x1, y1, x2, y2] in pixel coords
  float score_;
  int class_id_;
  int track_id_;
  int frame_id_;
  int lost_count_;
  bool is_activated_;
  TrajectoryState state_;

  STrack();

  /** Deep copy */
  STrack(const STrack &other);
  STrack &operator=(const STrack &other);

  /** Kalman predict step */
  void predict();

  /** Kalman update with measurement */
  virtual void update(const cv::Vec4f &ltrb, float score, int class_id);

  /** Activate a new track */
  void activate(int &cnt, int timestamp);

  /** Reactivate a lost track */
  void re_activate(const STrack &other, int &cnt, int timestamp);

  /** Convert ltrb to xyah (cx, cy, w/h, h) for Kalman measurement */
  cv::Vec4f ltrb_to_xyah() const;

  /** Convert xyah back to ltrb */
  static cv::Vec4f xyah_to_ltrb(const cv::Vec4f &xyah);

  /** Get current ltrb from Kalman state */
  cv::Vec4f get_ltrb_from_state() const;
};

/**
 * @brief ByteTrack: Multi-Object Tracking by Associating Every Detection Box
 *
 * Reference: https://github.com/ifzhang/ByteTrack
 * Paper: ByteTrack: Multi-Object Tracking by Associating Every Detection Box
 *
 * ByteTrack uses a simple two-stage matching strategy:
 *   1. Match high-score detections (score > track_thresh) with tracked tracks
 *   2. Match low-score detections with unmatched tracked tracks
 *   3. Match remaining tracked tracks with lost tracks
 *   4. Initialize new tracks from unmatched high-score detections
 */
class NNDEPLOY_CC_API ByteTrack {
 public:
  ByteTrack();
  virtual ~ByteTrack() {}

  /**
   * @brief Update tracker with detections
   * @param ltrb_boxes Vector of [x1, y1, x2, y2] bounding boxes (pixel coords)
   * @param scores Corresponding detection scores
   * @param class_ids Corresponding class IDs
   * @return MOTResult with tracked objects
   */
  virtual MOTResult update(const std::vector<cv::Vec4f> &ltrb_boxes,
                           const std::vector<float> &scores,
                           const std::vector<int> &class_ids);

  virtual void setTrackThresh(float thresh) { track_thresh_ = thresh; }
  virtual void setHighThresh(float thresh) { high_thresh_ = thresh; }
  virtual void setMatchThresh(float thresh) { match_thresh_ = thresh; }
  virtual void setMaxLostTime(int time) { max_lost_time_ = time; }
  virtual void setFrameRate(int fps) { frame_rate_ = fps; }

  /** Clear all tracks and reset state */
  virtual void reset();

 private:
  /** Compute IoU between two boxes [x1, y1, x2, y2] */
  static float iou(const cv::Vec4f &a, const cv::Vec4f &b);

  /** Compute IoU distance matrix (1 - IoU) between detections and tracks */
  static std::vector<std::vector<float>> iou_distance(
      const std::vector<STrack *> &tracks,
      const std::vector<cv::Vec4f> &detections);

  /** Hungarian-style linear assignment on cost matrix */
  static void linear_assignment(
      const std::vector<std::vector<float>> &cost_matrix,
      float cost_thresh, std::vector<std::pair<int, int>> &matches,
      std::vector<int> &unmatched_a, std::vector<int> &unmatched_b);

 protected:
  /** Predict all active tracks forward */
  virtual void multi_predict();

  /**
   * @brief Two-stage association
   * @return Matched pairs (track_idx, det_idx), unmatched tracks, unmatched dets
   */
  virtual void associate(const std::vector<STrack *> &active_tracks,
                         const std::vector<cv::Vec4f> &detections,
                         const std::vector<float> &scores,
                         const std::vector<int> &class_ids,
                         std::vector<std::tuple<int, int, float>> &matches,
                         std::vector<int> &unmatched_tracks,
                         std::vector<int> &unmatched_detections);

  std::vector<STrack> tracked_stracks_;
  std::vector<STrack> lost_stracks_;
  std::vector<STrack> removed_stracks_;

  int frame_id_;
  int track_id_counter_;

  float track_thresh_;    // Score threshold for first association (default 0.5)
  float high_thresh_;     // Score threshold for high-score detections (default 0.6)
  float match_thresh_;    // IoU threshold for matching (default 0.8)
  int max_lost_time_;     // Frames to keep lost tracks (default 30)
  int frame_rate_;        // Frame rate for time estimation (default 30)
};

}  // namespace track
}  // namespace nndeploy

#endif  // _NNDEPLOY_TRACK_BYTETRACK_BYTETRACK_H_
