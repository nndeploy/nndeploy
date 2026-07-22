#ifndef _NNDEPLOY_TRACK_BOXMOT_CONVERT_H_
#define _NNDEPLOY_TRACK_BOXMOT_CONVERT_H_

#include "nndeploy/base/common.h"
#include "nndeploy/base/macro.h"
#include "nndeploy/base/status.h"
#include "nndeploy/detect/result.h"
#include "nndeploy/detect/yolo_obb/result.h"
#include "nndeploy/track/boxmot/result.h"
#include "nndeploy/track/result.h"

namespace nndeploy {
namespace track {

/**
 * @brief Convert nndeploy BBoxResult to per-tracker Detection vector (AABB)
 *
 * Template works with bytetrack::Detection, botsort::Detection,
 * ocsort::Detection, sfsort::Detection, occluboost::Detection.
 * botsort/occluboost Detection has an extra embedding field (default empty).
 */
template <typename DetectionT>
std::vector<DetectionT> bboxResultToDetections(
    const detect::BBoxResult* bbox_result);

/**
 * @brief Convert nndeploy ObbResult to per-tracker Detection vector (OBB)
 */
template <typename DetectionT>
std::vector<DetectionT> obbResultToDetections(
    const detect::ObbResult* obb_result);

/**
 * @brief Convert per-tracker TrackOutput vector to nndeploy MOTResult
 */
template <typename TrackOutputT>
MOTResult trackOutputToMOTResult(const std::vector<TrackOutputT>& tracks);

/**
 * @brief Convert per-tracker TrackOutput vector to nndeploy BoxMotResult
 */
template <typename TrackOutputT>
BoxMotResult trackOutputToBoxMotResult(const std::vector<TrackOutputT>& tracks);

// ---------------------------------------------------------------------------
// Template implementations (must be in header for linking)
// ---------------------------------------------------------------------------

template <typename DetectionT>
std::vector<DetectionT> bboxResultToDetections(
    const detect::BBoxResult* bbox_result) {
  std::vector<DetectionT> detections;
  if (!bbox_result) return detections;
  detections.reserve(bbox_result->bboxs_.size());
  for (const auto& bbox : bbox_result->bboxs_) {
    DetectionT det;
    det.is_obb = false;
    // xyxy: xmin, ymin, xmax, ymax
    det.xyxy << bbox.bbox_[0], bbox.bbox_[1], bbox.bbox_[2], bbox.bbox_[3];
    // xywha: cx, cy, w, h, angle (0 for AABB)
    float cx = (bbox.bbox_[0] + bbox.bbox_[2]) * 0.5;
    float cy = (bbox.bbox_[1] + bbox.bbox_[3]) * 0.5;
    float w = bbox.bbox_[2] - bbox.bbox_[0];
    float h = bbox.bbox_[3] - bbox.bbox_[1];
    det.xywha << cx, cy, w, h, 0.0;
    det.conf = bbox.score_;
    det.cls = bbox.label_id_;
    det.det_ind = static_cast<int>(detections.size());
    detections.push_back(det);
  }
  return detections;
}

template <typename DetectionT>
std::vector<DetectionT> obbResultToDetections(
    const detect::ObbResult* obb_result) {
  std::vector<DetectionT> detections;
  if (!obb_result) return detections;
  detections.reserve(obb_result->boxes_.size());
  for (const auto& box : obb_result->boxes_) {
    DetectionT det;
    det.is_obb = true;
    // xywha: cx, cy, w, h, angle
    det.xywha << box.cx_, box.cy_, box.w_, box.h_, box.angle_;
    // xyxy: compute axis-aligned bounding box from OBB
    float cos_a = std::cos(box.angle_);
    float sin_a = std::sin(box.angle_);
    float dx = box.w_ * 0.5f;
    float dy = box.h_ * 0.5f;
    // Four corners relative to center
    float x1 = box.cx_ + (-dx * cos_a + dy * sin_a);
    float y1 = box.cy_ + (-dx * sin_a - dy * cos_a);
    float x2 = box.cx_ + (dx * cos_a + dy * sin_a);
    float y2 = box.cy_ + (dx * sin_a - dy * cos_a);
    float x3 = box.cx_ + (dx * cos_a - dy * sin_a);
    float y3 = box.cy_ + (dx * sin_a + dy * cos_a);
    float x4 = box.cx_ + (-dx * cos_a - dy * sin_a);
    float y4 = box.cy_ + (-dx * sin_a + dy * cos_a);
    float xmin = std::min({x1, x2, x3, x4});
    float ymin = std::min({y1, y2, y3, y4});
    float xmax = std::max({x1, x2, x3, x4});
    float ymax = std::max({y1, y2, y3, y4});
    det.xyxy << xmin, ymin, xmax, ymax;
    det.conf = box.score_;
    det.cls = box.label_id_;
    det.det_ind = static_cast<int>(detections.size());
    detections.push_back(det);
  }
  return detections;
}

template <typename TrackOutputT>
MOTResult trackOutputToMOTResult(const std::vector<TrackOutputT>& tracks) {
  MOTResult result;
  result.boxes.reserve(tracks.size());
  result.ids.reserve(tracks.size());
  result.scores.reserve(tracks.size());
  result.class_ids.reserve(tracks.size());
  for (const auto& track : tracks) {
    int x1 = static_cast<int>(std::round(track.xyxy(0)));
    int y1 = static_cast<int>(std::round(track.xyxy(1)));
    int x2 = static_cast<int>(std::round(track.xyxy(2)));
    int y2 = static_cast<int>(std::round(track.xyxy(3)));
    result.boxes.push_back({x1, y1, x2, y2});
    result.ids.push_back(track.id);
    result.scores.push_back(static_cast<float>(track.conf));
    result.class_ids.push_back(static_cast<int>(track.cls));
  }
  return result;
}

template <typename TrackOutputT>
BoxMotResult trackOutputToBoxMotResult(
    const std::vector<TrackOutputT>& tracks) {
  BoxMotResult result;
  result.tracks_.reserve(tracks.size());
  for (const auto& track : tracks) {
    BoxMotTrack bt;
    bt.id_ = track.id;
    bt.is_obb_ = track.is_obb;
    bt.confidence_ = static_cast<float>(track.conf);
    bt.class_id_ = static_cast<int>(track.cls);
    bt.detection_index_ = track.det_ind;
    // xyxy: xmin, ymin, xmax, ymax
    bt.bbox_[0] = static_cast<float>(track.xyxy(0));
    bt.bbox_[1] = static_cast<float>(track.xyxy(1));
    bt.bbox_[2] = static_cast<float>(track.xyxy(2));
    bt.bbox_[3] = static_cast<float>(track.xyxy(3));
    // xywha: cx, cy, w, h, angle
    bt.obb_[0] = static_cast<float>(track.xywha(0));
    bt.obb_[1] = static_cast<float>(track.xywha(1));
    bt.obb_[2] = static_cast<float>(track.xywha(2));
    bt.obb_[3] = static_cast<float>(track.xywha(3));
    bt.obb_[4] = static_cast<float>(track.xywha(4));
    result.tracks_.push_back(bt);
  }
  return result;
}

}  // namespace track
}  // namespace nndeploy

#endif  // _NNDEPLOY_TRACK_BOXMOT_CONVERT_H_
