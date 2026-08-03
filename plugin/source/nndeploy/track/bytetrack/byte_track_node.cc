#include "nndeploy/track/bytetrack/byte_track_node.h"

namespace nndeploy {
namespace track {

base::Status ByteTrackParam::serialize(rapidjson::Value &json,
                                        rapidjson::Document::AllocatorType &allocator) {
  json.AddMember("track_thresh_", track_thresh_, allocator);
  json.AddMember("high_thresh_", high_thresh_, allocator);
  json.AddMember("match_thresh_", match_thresh_, allocator);
  json.AddMember("max_lost_time_", max_lost_time_, allocator);
  json.AddMember("frame_rate_", frame_rate_, allocator);
  return base::kStatusCodeOk;
}

base::Status ByteTrackParam::deserialize(rapidjson::Value &json) {
  if (json.HasMember("track_thresh_"))
    track_thresh_ = json["track_thresh_"].GetFloat();
  if (json.HasMember("high_thresh_"))
    high_thresh_ = json["high_thresh_"].GetFloat();
  if (json.HasMember("match_thresh_"))
    match_thresh_ = json["match_thresh_"].GetFloat();
  if (json.HasMember("max_lost_time_"))
    max_lost_time_ = json["max_lost_time_"].GetInt();
  if (json.HasMember("frame_rate_"))
    frame_rate_ = json["frame_rate_"].GetInt();
  return base::kStatusCodeOk;
}

base::Status ByteTrackNode::init() {
  ByteTrackParam *param = (ByteTrackParam *)param_.get();
  tracker_ = std::make_unique<ByteTrack>();
  tracker_->setTrackThresh(param->track_thresh_);
  tracker_->setHighThresh(param->high_thresh_);
  tracker_->setMatchThresh(param->match_thresh_);
  tracker_->setMaxLostTime(param->max_lost_time_);
  tracker_->setFrameRate(param->frame_rate_);
  return base::kStatusCodeOk;
}

base::Status ByteTrackNode::deinit() {
  tracker_.reset();
  return base::kStatusCodeOk;
}

base::Status ByteTrackNode::run() {
  detect::BBoxResult *bbox_result =
      (detect::BBoxResult *)inputs_[0]->getParam(this);
  if (bbox_result == nullptr) {
    NNDEPLOY_LOGE("ByteTrackNode: input BBoxResult is null\n");
    return base::kStatusCodeErrorInvalidValue;
  }

  // Convert BBoxResult to ByteTrack input format
  std::vector<cv::Vec4f> ltrb_boxes;
  std::vector<float> scores;
  std::vector<int> class_ids;

  for (const auto &bbox : bbox_result->bboxs_) {
    ltrb_boxes.push_back(cv::Vec4f(bbox.bbox_[0], bbox.bbox_[1],
                                    bbox.bbox_[2], bbox.bbox_[3]));
    scores.push_back(bbox.score_);
    class_ids.push_back(bbox.label_id_);
  }

  // Run tracking
  MOTResult *mot_result = new MOTResult();
  *mot_result = tracker_->update(ltrb_boxes, scores, class_ids);

  outputs_[0]->set(mot_result, false);
  return base::kStatusCodeOk;
}

REGISTER_NODE("nndeploy::track::ByteTrackNode", ByteTrackNode);

}  // namespace track
}  // namespace nndeploy
