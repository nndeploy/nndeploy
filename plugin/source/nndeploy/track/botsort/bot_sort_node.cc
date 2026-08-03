#include "nndeploy/track/botsort/bot_sort_node.h"

namespace nndeploy {
namespace track {

base::Status BotSortNode::init() {
  tracker_ = std::make_unique<BotSORT>();
  return base::kStatusCodeOk;
}

base::Status BotSortNode::deinit() {
  tracker_.reset();
  return base::kStatusCodeOk;
}

base::Status BotSortNode::run() {
  // Input 0: cv::Mat (video frame, for GMC)
  cv::Mat *img = inputs_[0]->getCvMat(this);
  if (img == nullptr) {
    NNDEPLOY_LOGE("BotSortNode: input frame is null\n");
    return base::kStatusCodeErrorInvalidValue;
  }

  // Input 1: BBoxResult
  detect::BBoxResult *bbox_result =
      (detect::BBoxResult *)inputs_[1]->getParam(this);
  if (bbox_result == nullptr) {
    NNDEPLOY_LOGE("BotSortNode: input BBoxResult is null\n");
    return base::kStatusCodeErrorInvalidValue;
  }

  // Convert BBoxResult to ByteTrack/BotSORT input format
  std::vector<cv::Vec4f> ltrb_boxes;
  std::vector<float> scores;
  std::vector<int> class_ids;

  for (const auto &bbox : bbox_result->bboxs_) {
    ltrb_boxes.push_back(cv::Vec4f(bbox.bbox_[0], bbox.bbox_[1],
                                    bbox.bbox_[2], bbox.bbox_[3]));
    scores.push_back(bbox.score_);
    class_ids.push_back(bbox.label_id_);
  }

  // Provide current frame for GMC, then run tracking
  tracker_->setFrame(*img);
  MOTResult *mot_result = new MOTResult();
  *mot_result = tracker_->update(ltrb_boxes, scores, class_ids);

  outputs_[0]->set(mot_result, false);
  return base::kStatusCodeOk;
}

REGISTER_NODE("nndeploy::track::BotSortNode", BotSortNode);

}  // namespace track
}  // namespace nndeploy
