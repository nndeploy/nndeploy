#include "nndeploy/track/botsort/botsort.h"

#include <algorithm>
#include <cmath>

#include <opencv2/video/tracking.hpp>  // estimateAffinePartial2D
#include <opencv2/calib3d.hpp>         // cv::RANSAC

#include "nndeploy/base/log.h"

namespace nndeploy {
namespace track {

BotSORT::BotSORT()
    : ByteTrack(),
      gm_scale_(0.2f) {
  orb_ = cv::ORB::create(1000);
}

void BotSORT::setFrame(const cv::Mat &img) {
  curr_frame_ = img.clone();
}

void BotSORT::multi_predict() {
  // Apply GMC before predicting, if we have a frame and tracks
  if (!curr_frame_.empty() && !tracked_stracks_.empty()) {
    cv::Mat affine = compute_gmc(curr_frame_);

    // Check for significant motion (not identity)
    bool has_motion = false;
    for (int r = 0; r < 2 && !has_motion; ++r) {
      for (int c = 0; c < 3 && !has_motion; ++c) {
        float expected = (r == c) ? 1.0f : 0.0f;
        if (std::abs(affine.at<float>(r, c) - expected) > 0.001f) {
          has_motion = true;
        }
      }
    }

    if (has_motion) {
      for (auto &track : tracked_stracks_) {
        apply_gmc_to_track(track, affine);
      }
    }
  }

  // Call base class multi_predict
  ByteTrack::multi_predict();
}

cv::Mat BotSORT::compute_gmc(const cv::Mat &curr_img) {
  if (prev_gray_.empty()) {
    if (curr_img.channels() == 3) {
      cv::cvtColor(curr_img, prev_gray_, cv::COLOR_BGR2GRAY);
    } else {
      prev_gray_ = curr_img.clone();
    }
    return cv::Mat::eye(2, 3, CV_32F);
  }

  cv::Mat curr_gray;
  if (curr_img.channels() == 3) {
    cv::cvtColor(curr_img, curr_gray, cv::COLOR_BGR2GRAY);
  } else {
    curr_gray = curr_img;
  }

  std::vector<cv::KeyPoint> prev_kpts, curr_kpts;
  cv::Mat prev_desc, curr_desc;
  orb_->detectAndCompute(prev_gray_, cv::noArray(), prev_kpts, prev_desc);
  orb_->detectAndCompute(curr_gray, cv::noArray(), curr_kpts, curr_desc);

  cv::Mat affine = cv::Mat::eye(2, 3, CV_32F);

  if (!prev_desc.empty() && !curr_desc.empty() && prev_kpts.size() > 10 &&
      curr_kpts.size() > 10) {
    cv::BFMatcher matcher(cv::NORM_HAMMING, true);
    std::vector<cv::DMatch> matches;
    matcher.match(prev_desc, curr_desc, matches);

    std::sort(matches.begin(), matches.end(),
              [](const cv::DMatch &a, const cv::DMatch &b) {
                return a.distance < b.distance;
              });

    int num_good = std::min((int)matches.size(),
                            std::max(10, (int)(matches.size() * gm_scale_)));
    if (num_good >= 4) {
      std::vector<cv::Point2f> prev_pts, curr_pts;
      for (int i = 0; i < num_good; ++i) {
        prev_pts.push_back(prev_kpts[matches[i].queryIdx].pt);
        curr_pts.push_back(curr_kpts[matches[i].trainIdx].pt);
      }

      affine = cv::estimateAffinePartial2D(prev_pts, curr_pts, cv::noArray(),
                                            cv::RANSAC, 3.0f);
      if (affine.empty()) {
        affine = cv::Mat::eye(2, 3, CV_32F);
      }
    }
  }

  prev_gray_ = curr_gray;
  return affine;
}

void BotSORT::apply_gmc_to_track(STrack &track, const cv::Mat &affine) {
  float cx = track.mean_.at<float>(0);
  float cy = track.mean_.at<float>(1);
  float w = track.mean_.at<float>(2);
  float h = track.mean_.at<float>(3);

  float new_cx = affine.at<float>(0, 0) * cx + affine.at<float>(0, 1) * cy +
                 affine.at<float>(0, 2);
  float new_cy = affine.at<float>(1, 0) * cx + affine.at<float>(1, 1) * cy +
                 affine.at<float>(1, 2);

  float scale_x = std::sqrt(affine.at<float>(0, 0) * affine.at<float>(0, 0) +
                             affine.at<float>(0, 1) * affine.at<float>(0, 1));
  float scale_y = std::sqrt(affine.at<float>(1, 0) * affine.at<float>(1, 0) +
                             affine.at<float>(1, 1) * affine.at<float>(1, 1));

  track.mean_.at<float>(0) = new_cx;
  track.mean_.at<float>(1) = new_cy;
  track.mean_.at<float>(2) = w * (scale_x / scale_y);
  track.mean_.at<float>(3) = h * scale_y;

  track.ltrb_ = track.get_ltrb_from_state();
}

MOTResult BotSORT::update(const std::vector<cv::Vec4f> &ltrb_boxes,
                          const std::vector<float> &scores,
                          const std::vector<int> &class_ids) {
  return ByteTrack::update(ltrb_boxes, scores, class_ids);
}

}  // namespace track
}  // namespace nndeploy
