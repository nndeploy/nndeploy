#include "nndeploy/track/bytetrack/bytetrack.h"

#include <algorithm>
#include <cmath>
#include <limits>

#include "nndeploy/base/log.h"

namespace nndeploy {
namespace track {

// ===== Kalman filter constants =====
static constexpr float kStdWeightPosition = 1.0f / 20.0f;
static constexpr float kStdWeightVelocity = 1.0f / 160.0f;

// ===== STrack implementation =====

STrack::STrack()
    : mean_(cv::Mat::zeros(8, 1, CV_32F)),
      covariance_(cv::Mat::eye(8, 8, CV_32F)),
      ltrb_(cv::Vec4f()),
      score_(0.0f),
      class_id_(-1),
      track_id_(-1),
      frame_id_(0),
      lost_count_(0),
      is_activated_(false),
      state_(New) {}

STrack::STrack(const STrack &other)
    : ltrb_(other.ltrb_),
      score_(other.score_),
      class_id_(other.class_id_),
      track_id_(other.track_id_),
      frame_id_(other.frame_id_),
      lost_count_(other.lost_count_),
      is_activated_(other.is_activated_),
      state_(other.state_) {
  other.mean_.copyTo(mean_);
  other.covariance_.copyTo(covariance_);
}

STrack &STrack::operator=(const STrack &other) {
  if (this != &other) {
    other.mean_.copyTo(mean_);
    other.covariance_.copyTo(covariance_);
    ltrb_ = other.ltrb_;
    score_ = other.score_;
    class_id_ = other.class_id_;
    track_id_ = other.track_id_;
    frame_id_ = other.frame_id_;
    lost_count_ = other.lost_count_;
    is_activated_ = other.is_activated_;
    state_ = other.state_;
  }
  return *this;
}

cv::Vec4f STrack::ltrb_to_xyah() const {
  cv::Vec4f xyah;
  float w = ltrb_[2] - ltrb_[0];
  float h = ltrb_[3] - ltrb_[1];
  xyah[0] = (ltrb_[0] + ltrb_[2]) * 0.5f;  // cx
  xyah[1] = (ltrb_[1] + ltrb_[3]) * 0.5f;  // cy
  xyah[2] = w / h;                           // aspect ratio
  xyah[3] = h;                               // height
  return xyah;
}

cv::Vec4f STrack::xyah_to_ltrb(const cv::Vec4f &xyah) {
  cv::Vec4f ltrb;
  float w = xyah[2] * xyah[3];  // aspect * height
  ltrb[0] = xyah[0] - w * 0.5f;
  ltrb[1] = xyah[1] - xyah[3] * 0.5f;
  ltrb[2] = xyah[0] + w * 0.5f;
  ltrb[3] = xyah[1] + xyah[3] * 0.5f;
  return ltrb;
}

cv::Vec4f STrack::get_ltrb_from_state() const {
  cv::Vec4f xyah;
  xyah[0] = mean_.at<float>(0);  // cx
  xyah[1] = mean_.at<float>(1);  // cy
  xyah[2] = mean_.at<float>(2);  // w (actually aspect ratio * h)
  xyah[3] = mean_.at<float>(3);  // h
  return xyah_to_ltrb(xyah);
}

void STrack::predict() {
  // Constant velocity model
  // mean = F * mean  where F = [I_4, I_4; 0, I_4]
  mean_.at<float>(0) += mean_.at<float>(4);
  mean_.at<float>(1) += mean_.at<float>(5);
  mean_.at<float>(2) += mean_.at<float>(6);
  mean_.at<float>(3) += mean_.at<float>(7);

  // covariance = F * covariance * F^T + Q
  // F is: [I, I; 0, I] so F * cov * F^T shifts the covariance
  cv::Mat F = cv::Mat::eye(8, 8, CV_32F);
  F.at<float>(0, 4) = 1.0f;
  F.at<float>(1, 5) = 1.0f;
  F.at<float>(2, 6) = 1.0f;
  F.at<float>(3, 7) = 1.0f;

  covariance_ = F * covariance_ * F.t();

  // Process noise Q
  float sp = kStdWeightPosition;
  float sv = kStdWeightVelocity;
  cv::Mat Q = cv::Mat::zeros(8, 8, CV_32F);
  for (int i = 0; i < 4; ++i) {
    Q.at<float>(i, i) = sp * sp;
    Q.at<float>(i + 4, i + 4) = sv * sv;
  }
  covariance_ += Q;

  // Update ltrb from predicted state
  ltrb_ = get_ltrb_from_state();
}

void STrack::update(const cv::Vec4f &ltrb, float score, int class_id) {
  score_ = score;
  class_id_ = class_id_;
  ltrb_ = ltrb;

  // Measurement: [cx, cy, aspect, height]
  cv::Vec4f xyah = ltrb_to_xyah();
  cv::Mat z(4, 1, CV_32F);
  z.at<float>(0) = xyah[0];
  z.at<float>(1) = xyah[1];
  z.at<float>(2) = xyah[2];
  z.at<float>(3) = xyah[3];

  // H = [I_4, 0_4]
  cv::Mat H(4, 8, CV_32F, cv::Scalar(0.0f));
  H.at<float>(0, 0) = 1.0f;
  H.at<float>(1, 1) = 1.0f;
  H.at<float>(2, 2) = 1.0f;
  H.at<float>(3, 3) = 1.0f;

  // Measurement noise covariance R
  cv::Mat R = cv::Mat::eye(4, 4, CV_32F) * 0.1f;

  // Innovation: y = z - H * mean
  cv::Mat H_mean = H * mean_;
  cv::Mat y = z - H_mean;

  // Innovation covariance: S = H * P * H^T + R
  cv::Mat S = H * covariance_ * H.t() + R;

  // Kalman gain: K = P * H^T * S^(-1)
  cv::Mat K = covariance_ * H.t() * S.inv();

  // Update: mean = mean + K * y
  mean_ = mean_ + K * y;

  // Update: P = (I - K * H) * P
  cv::Mat I = cv::Mat::eye(8, 8, CV_32F);
  covariance_ = (I - K * H) * covariance_;

  // Update ltrb from corrected state
  ltrb_ = get_ltrb_from_state();
}

void STrack::activate(int &cnt, int timestamp) {
  // Initialize Kalman filter from detection
  cv::Vec4f xyah = ltrb_to_xyah();
  mean_.at<float>(0) = xyah[0];  // cx
  mean_.at<float>(1) = xyah[1];  // cy
  mean_.at<float>(2) = xyah[2];  // aspect ratio
  mean_.at<float>(3) = xyah[3];  // height
  mean_.at<float>(4) = 0.0f;     // vx
  mean_.at<float>(5) = 0.0f;     // vy
  mean_.at<float>(6) = 0.0f;     // vw (aspect velocity)
  mean_.at<float>(7) = 0.0f;     // vh (height velocity)

  // Initialize covariance with uncertainty
  float sp = kStdWeightPosition;
  float sv = kStdWeightVelocity;
  for (int i = 0; i < 8; ++i) {
    float val = (i < 4) ? sp * sp : sv * sv;
    covariance_.at<float>(i, i) = val * val;
  }

  track_id_ = ++cnt;
  frame_id_ = timestamp;
  lost_count_ = 0;
  is_activated_ = true;
  state_ = Tracked;
}

void STrack::re_activate(const STrack &other, int &cnt, int timestamp) {
  // Use the new detection to reset Kalman state
  cv::Vec4f xyah = other.ltrb_to_xyah();
  mean_.at<float>(0) = xyah[0];
  mean_.at<float>(1) = xyah[1];
  mean_.at<float>(2) = xyah[2];
  mean_.at<float>(3) = xyah[3];
  mean_.at<float>(4) = 0.0f;
  mean_.at<float>(5) = 0.0f;
  mean_.at<float>(6) = 0.0f;
  mean_.at<float>(7) = 0.0f;

  ltrb_ = other.ltrb_;
  score_ = other.score_;
  class_id_ = other.class_id_;
  frame_id_ = timestamp;
  lost_count_ = 0;
  is_activated_ = true;
  state_ = Tracked;
}

// ===== ByteTrack implementation =====

ByteTrack::ByteTrack()
    : frame_id_(0),
      track_id_counter_(0),
      track_thresh_(0.5f),
      high_thresh_(0.6f),
      match_thresh_(0.8f),
      max_lost_time_(30),
      frame_rate_(30) {}

void ByteTrack::reset() {
  tracked_stracks_.clear();
  lost_stracks_.clear();
  removed_stracks_.clear();
  frame_id_ = 0;
  track_id_counter_ = 0;
}

float ByteTrack::iou(const cv::Vec4f &a, const cv::Vec4f &b) {
  float ax1 = a[0], ay1 = a[1], ax2 = a[2], ay2 = a[3];
  float bx1 = b[0], by1 = b[1], bx2 = b[2], by2 = b[3];

  float inter_x1 = std::max(ax1, bx1);
  float inter_y1 = std::max(ay1, by1);
  float inter_x2 = std::min(ax2, bx2);
  float inter_y2 = std::min(ay2, by2);

  float inter_w = std::max(0.0f, inter_x2 - inter_x1);
  float inter_h = std::max(0.0f, inter_y2 - inter_y1);
  float inter_area = inter_w * inter_h;

  float area_a = (ax2 - ax1) * (ay2 - ay1);
  float area_b = (bx2 - bx1) * (by2 - by1);
  float union_area = area_a + area_b - inter_area;

  if (union_area <= 0.0f) return 0.0f;
  return inter_area / union_area;
}

std::vector<std::vector<float>> ByteTrack::iou_distance(
    const std::vector<STrack *> &tracks,
    const std::vector<cv::Vec4f> &detections) {
  int n = (int)tracks.size();
  int m = (int)detections.size();
  std::vector<std::vector<float>> cost(n, std::vector<float>(m, 0.0f));

  for (int i = 0; i < n; ++i) {
    cv::Vec4f track_ltrb = tracks[i]->ltrb_;
    for (int j = 0; j < m; ++j) {
      float iou_val = iou(track_ltrb, detections[j]);
      cost[i][j] = 1.0f - iou_val;  // distance = 1 - IoU
    }
  }
  return cost;
}

void ByteTrack::linear_assignment(
    const std::vector<std::vector<float>> &cost_matrix, float cost_thresh,
    std::vector<std::pair<int, int>> &matches,
    std::vector<int> &unmatched_a, std::vector<int> &unmatched_b) {
  int n = (int)cost_matrix.size();
  int m = (n > 0) ? (int)cost_matrix[0].size() : 0;

  std::vector<bool> matched_a(n, false);
  std::vector<bool> matched_b(m, false);

  // Greedy matching: for each track, find the best unmatched detection
  for (int i = 0; i < n; ++i) {
    float best_cost = cost_thresh;
    int best_j = -1;
    for (int j = 0; j < m; ++j) {
      if (!matched_b[j] && cost_matrix[i][j] < best_cost) {
        best_cost = cost_matrix[i][j];
        best_j = j;
      }
    }
    if (best_j >= 0) {
      matches.emplace_back(i, best_j);
      matched_a[i] = true;
      matched_b[best_j] = true;
    }
  }

  // Collect unmatched
  for (int i = 0; i < n; ++i) {
    if (!matched_a[i]) unmatched_a.push_back(i);
  }
  for (int j = 0; j < m; ++j) {
    if (!matched_b[j]) unmatched_b.push_back(j);
  }
}

void ByteTrack::multi_predict() {
  for (auto &track : tracked_stracks_) {
    track.predict();
  }
}

void ByteTrack::associate(
    const std::vector<STrack *> &active_tracks,
    const std::vector<cv::Vec4f> &detections,
    const std::vector<float> &scores,
    const std::vector<int> &class_ids,
    std::vector<std::tuple<int, int, float>> &matches,
    std::vector<int> &unmatched_tracks,
    std::vector<int> &unmatched_detections) {
  (void)scores;
  (void)class_ids;

  if (active_tracks.empty() || detections.empty()) {
    unmatched_tracks.resize(active_tracks.size());
    for (int i = 0; i < (int)active_tracks.size(); ++i) {
      unmatched_tracks[i] = i;
    }
    unmatched_detections.resize(detections.size());
    for (int j = 0; j < (int)detections.size(); ++j) {
      unmatched_detections[j] = j;
    }
    return;
  }

  // Compute IoU distance matrix
  auto cost_matrix = iou_distance(active_tracks, detections);

  // Linear assignment
  std::vector<std::pair<int, int>> raw_matches;
  std::vector<int> raw_unmatched_a, raw_unmatched_b;
  linear_assignment(cost_matrix, 1.0f - match_thresh_, raw_matches,
                    raw_unmatched_a, raw_unmatched_b);

  // Filter matches by actual cost threshold
  for (auto &m : raw_matches) {
    int ti = m.first;
    int dj = m.second;
    float cost = cost_matrix[ti][dj];
    if (cost < 1.0f - match_thresh_) {
      matches.emplace_back(ti, dj, cost);
    } else {
      unmatched_tracks.push_back(ti);
      unmatched_detections.push_back(dj);
    }
  }

  unmatched_tracks.insert(unmatched_tracks.end(), raw_unmatched_a.begin(),
                           raw_unmatched_a.end());
  unmatched_detections.insert(unmatched_detections.end(),
                               raw_unmatched_b.begin(),
                               raw_unmatched_b.end());
}

MOTResult ByteTrack::update(const std::vector<cv::Vec4f> &ltrb_boxes,
                            const std::vector<float> &scores,
                            const std::vector<int> &class_ids) {
  ++frame_id_;

  // Separate high and low score detections
  std::vector<cv::Vec4f> high_dets, low_dets;
  std::vector<float> high_scores, low_scores;
  std::vector<int> high_class_ids, low_class_ids;

  for (int i = 0; i < (int)ltrb_boxes.size(); ++i) {
    if (scores[i] >= track_thresh_) {
      high_dets.push_back(ltrb_boxes[i]);
      high_scores.push_back(scores[i]);
      high_class_ids.push_back(class_ids[i]);
    } else {
      low_dets.push_back(ltrb_boxes[i]);
      low_scores.push_back(scores[i]);
      low_class_ids.push_back(class_ids[i]);
    }
  }

  // Predict current tracks
  multi_predict();

  // Step 1: Filter out tracks not in "Tracked" state
  std::vector<STrack *> active_tracks;
  for (auto &track : tracked_stracks_) {
    if (track.state_ == Tracked) {
      active_tracks.push_back(&track);
    }
  }

  // Step 2: First association — high score detections with tracked tracks
  std::vector<std::tuple<int, int, float>> match_pairs;
  std::vector<int> unmatched_tracks_first, unmatched_dets_first;
  associate(active_tracks, high_dets, high_scores, high_class_ids,
            match_pairs, unmatched_tracks_first, unmatched_dets_first);

  // Apply first association matches
  std::vector<bool> is_matched_det(high_dets.size(), false);
  for (auto &mp : match_pairs) {
    int ti = std::get<0>(mp);
    int dj = std::get<1>(mp);
    active_tracks[ti]->update(high_dets[dj], high_scores[dj],
                               high_class_ids[dj]);
    is_matched_det[dj] = true;
  }

  // Step 3: Second association — remaining tracked tracks with unmatched
  // high-score detections via IoU, then low-score detections
  std::vector<STrack *> rem_tracks;
  for (auto idx : unmatched_tracks_first) {
    rem_tracks.push_back(active_tracks[idx]);
  }

  // Collect remaining unmatched high-score detections
  std::vector<cv::Vec4f> rem_high_dets;
  std::vector<float> rem_high_scores;
  std::vector<int> rem_high_class_ids;
  for (int j = 0; j < (int)high_dets.size(); ++j) {
    if (!is_matched_det[j]) {
      rem_high_dets.push_back(high_dets[j]);
      rem_high_scores.push_back(high_scores[j]);
      rem_high_class_ids.push_back(high_class_ids[j]);
    }
  }

  // Try to match remaining tracks with remaining high-score detections
  std::vector<std::tuple<int, int, float>> second_match_pairs;
  std::vector<int> unmatched_tracks_second, unmatched_high_dets_second;
  associate(rem_tracks, rem_high_dets, rem_high_scores, rem_high_class_ids,
            second_match_pairs, unmatched_tracks_second,
            unmatched_high_dets_second);

  // Apply second association matches
  std::vector<bool> is_matched_low(rem_high_dets.size(), false);
  for (auto &mp : second_match_pairs) {
    int ti = std::get<0>(mp);
    int dj = std::get<1>(mp);
    rem_tracks[ti]->update(rem_high_dets[dj], rem_high_scores[dj],
                            rem_high_class_ids[dj]);
    is_matched_low[dj] = true;
  }

  // Step 4: Try to match remaining unmatched tracks with low-score detections
  std::vector<STrack *> remaining_tracks;
  for (auto idx : unmatched_tracks_second) {
    remaining_tracks.push_back(rem_tracks[idx]);
  }

  std::vector<std::tuple<int, int, float>> low_match_pairs;
  std::vector<int> unmatched_tracks_low, unmatched_low_dets;
  associate(remaining_tracks, low_dets, low_scores, low_class_ids,
            low_match_pairs, unmatched_tracks_low, unmatched_low_dets);

  for (auto &mp : low_match_pairs) {
    int ti = std::get<0>(mp);
    int dj = std::get<1>(mp);
    remaining_tracks[ti]->update(low_dets[dj], low_scores[dj],
                                  low_class_ids[dj]);
  }

  // Mark unmatched remaining tracks as lost
  for (auto idx : unmatched_tracks_low) {
    remaining_tracks[idx]->state_ = Lost;
    remaining_tracks[idx]->lost_count_ = 0;
    lost_stracks_.push_back(*remaining_tracks[idx]);
  }

  // Step 5: Match unmatched (and newly lost) tracks with lost tracks via IoU
  if (!lost_stracks_.empty()) {
    // Build pool of unmatched track pointers
    std::vector<STrack *> unmatched_track_ptrs;
    for (auto idx : unmatched_tracks_low) {
      unmatched_track_ptrs.push_back(remaining_tracks[idx]);
    }

    // Get lost tracks that are still within max_lost_time
    std::vector<STrack *> active_lost;
    for (auto &lost : lost_stracks_) {
      if (lost.lost_count_ < max_lost_time_) {
        active_lost.push_back(&lost);
      }
    }

    if (!unmatched_track_ptrs.empty() && !active_lost.empty()) {
      // Compute IoU between unmatched tracks and lost tracks
      std::vector<cv::Vec4f> lost_ltrb;
      for (auto *l : active_lost) {
        lost_ltrb.push_back(l->ltrb_);
      }

      // Build track->track IoU cost matrix
      int n = (int)unmatched_track_ptrs.size();
      int m = (int)active_lost.size();
      std::vector<std::vector<float>> cost(
          n, std::vector<float>(m, 0.0f));
      for (int i = 0; i < n; ++i) {
        for (int j = 0; j < m; ++j) {
          cost[i][j] = 1.0f - iou(unmatched_track_ptrs[i]->ltrb_,
                                   active_lost[j]->ltrb_);
        }
      }

      std::vector<std::pair<int, int>> lost_matches;
      std::vector<int> unmatched_lost_a, unmatched_lost_b;
      linear_assignment(cost, 1.0f - match_thresh_, lost_matches,
                        unmatched_lost_a, unmatched_lost_b);

      // Reactivate matched lost tracks
      for (auto &lm : lost_matches) {
        int ti = lm.first;
        int lj = lm.second;
        active_lost[lj]->re_activate(*unmatched_track_ptrs[ti],
                                      track_id_counter_, frame_id_);
      }
    }
  }

  // Step 6: Initialize new tracks from unmatched high-score detections
  for (int j = 0; j < (int)high_dets.size(); ++j) {
    if (!is_matched_det[j] && high_scores[j] >= high_thresh_) {
      // Also check second association unmatched
      bool already_matched = false;
      for (auto &mp : second_match_pairs) {
        if (std::get<1>(mp) == j) {
          already_matched = true;
          break;
        }
      }
      if (!already_matched) {
        STrack new_track;
        new_track.ltrb_ = high_dets[j];
        new_track.score_ = high_scores[j];
        new_track.class_id_ = high_class_ids[j];
        new_track.activate(track_id_counter_, frame_id_);
        tracked_stracks_.push_back(new_track);
      }
    }
  }

  // Step 7: Update lost tracks counter and remove expired
  for (auto &lost : lost_stracks_) {
    lost.lost_count_++;
  }

  // Remove expired lost tracks
  std::vector<STrack> active_lost_tracks;
  for (auto &lost : lost_stracks_) {
    if (lost.lost_count_ <= max_lost_time_) {
      active_lost_tracks.push_back(lost);
    } else {
      removed_stracks_.push_back(lost);
    }
  }
  lost_stracks_ = active_lost_tracks;

  // Step 8: Clean up tracked_stracks_ — remove non-tracked, add back tracked
  // lost tracks
  std::vector<STrack> active_tracked;
  for (auto &track : tracked_stracks_) {
    if (track.state_ == Tracked) {
      active_tracked.push_back(track);
    } else {
      // Already handled above — these were moved to lost
    }
  }
  tracked_stracks_ = active_tracked;

  // Also add back any lost tracks that might have been re-activated
  // (they should already be in tracked_stracks_ via the lost->tracked
  // promotion in the lost matching step, but since we use value semantics
  // the reactivated lost tracks are updated in place in lost_stracks_
  // and need to be moved back)
  std::vector<STrack> still_lost;
  for (auto &lost : lost_stracks_) {
    if (lost.state_ == Tracked) {
      // This track was re-activated, move it back to tracked
      tracked_stracks_.push_back(lost);
    } else {
      still_lost.push_back(lost);
    }
  }
  lost_stracks_ = still_lost;

  // Step 9: Build output
  MOTResult result;
  for (auto &track : tracked_stracks_) {
    if (track.state_ == Tracked && track.is_activated_) {
      cv::Vec4f ltrb = track.ltrb_;
      result.boxes.push_back({(int)ltrb[0], (int)ltrb[1], (int)ltrb[2],
                              (int)ltrb[3]});
      result.ids.push_back(track.track_id_);
      result.scores.push_back(track.score_);
      result.class_ids.push_back(track.class_id_);
    }
  }

  return result;
}

}  // namespace track
}  // namespace nndeploy
