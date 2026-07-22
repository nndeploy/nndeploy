#include "nndeploy/track/boxmot/boxmot_node.h"

// BoxMot native C++ tracker headers
#include "botsort/tracker.hpp"
#include "botsort/types.hpp"
#include "bytetrack/tracker.hpp"
#include "bytetrack/types.hpp"
#include "nndeploy/track/boxmot/boxmot_convert.h"
#include "occluboost/tracker.hpp"
#include "occluboost/types.hpp"
#include "ocsort/tracker.hpp"
#include "ocsort/types.hpp"
#include "sfsort/tracker.hpp"
#include "sfsort/types.hpp"

namespace nndeploy {
namespace track {

// =========================================================================
// init: create tracker based on tracker_type_
// =========================================================================

base::Status BoxMotNode::init() {
  BoxMotParam* param = static_cast<BoxMotParam*>(param_.get());
  active_tracker_type_ = param->tracker_type_;

  switch (active_tracker_type_) {
    case kTrackerTypeByteTrack: {
      bytetrack::Config cfg;
      cfg.min_conf = param->bytetrack_param_.min_conf_;
      cfg.track_thresh = param->bytetrack_param_.track_thresh_;
      cfg.match_thresh = param->bytetrack_param_.match_thresh_;
      cfg.track_buffer = param->bytetrack_param_.track_buffer_;
      cfg.frame_rate = param->bytetrack_param_.frame_rate_;
      cfg.max_obs = param->bytetrack_param_.max_obs_;
      tracker_ = new bytetrack::ByteTrackTracker(cfg);
      break;
    }
    case kTrackerTypeBotSort: {
      botsort::Config cfg;
      cfg.track_high_thresh = param->botsort_param_.track_high_thresh_;
      cfg.track_low_thresh = param->botsort_param_.track_low_thresh_;
      cfg.new_track_thresh = param->botsort_param_.new_track_thresh_;
      cfg.track_buffer = param->botsort_param_.track_buffer_;
      cfg.match_thresh = param->botsort_param_.match_thresh_;
      cfg.proximity_thresh = param->botsort_param_.proximity_thresh_;
      cfg.appearance_thresh = param->botsort_param_.appearance_thresh_;
      cfg.cmc_method = param->botsort_param_.cmc_method_;
      cfg.frame_rate = param->botsort_param_.frame_rate_;
      cfg.fuse_first_associate = param->botsort_param_.fuse_first_associate_;
      cfg.with_reid = param->botsort_param_.with_reid_;
      cfg.max_obs = param->botsort_param_.max_obs_;
      cfg.reid_preprocess = param->botsort_param_.reid_preprocess_;
      if (!param->reid_param_.reid_model_path_.empty()) {
        cfg.reid_model_path = param->reid_param_.reid_model_path_;
      }
      tracker_ = new botsort::BotSortTracker(cfg);
      break;
    }
    case kTrackerTypeOcSort: {
      ocsort::Config cfg;
      cfg.min_conf = param->ocsort_param_.min_conf_;
      cfg.det_thresh = param->ocsort_param_.det_thresh_;
      cfg.iou_threshold = param->ocsort_param_.iou_threshold_;
      cfg.max_age = param->ocsort_param_.max_age_;
      cfg.min_hits = param->ocsort_param_.min_hits_;
      cfg.delta_t = param->ocsort_param_.delta_t_;
      cfg.use_byte = param->ocsort_param_.use_byte_;
      cfg.inertia = param->ocsort_param_.inertia_;
      cfg.q_xy_scaling = param->ocsort_param_.q_xy_scaling_;
      cfg.q_s_scaling = param->ocsort_param_.q_s_scaling_;
      cfg.max_obs = param->ocsort_param_.max_obs_;
      tracker_ = new ocsort::OCSORTTracker(cfg);
      break;
    }
    case kTrackerTypeSfSort: {
      sfsort::Config cfg;
      cfg.high_th = param->sfsort_param_.high_th_;
      cfg.match_th_first = param->sfsort_param_.match_th_first_;
      cfg.new_track_th = param->sfsort_param_.new_track_th_;
      cfg.low_th = param->sfsort_param_.low_th_;
      cfg.match_th_second = param->sfsort_param_.match_th_second_;
      cfg.dynamic_tuning = param->sfsort_param_.dynamic_tuning_;
      cfg.cth = param->sfsort_param_.cth_;
      cfg.high_th_m = param->sfsort_param_.high_th_m_;
      cfg.new_track_th_m = param->sfsort_param_.new_track_th_m_;
      cfg.match_th_first_m = param->sfsort_param_.match_th_first_m_;
      cfg.obb_theta_damping = param->sfsort_param_.obb_theta_damping_;
      cfg.marginal_timeout = param->sfsort_param_.marginal_timeout_;
      cfg.central_timeout = param->sfsort_param_.central_timeout_;
      cfg.frame_width = param->sfsort_param_.frame_width_;
      cfg.frame_height = param->sfsort_param_.frame_height_;
      cfg.horizontal_margin = param->sfsort_param_.horizontal_margin_;
      cfg.vertical_margin = param->sfsort_param_.vertical_margin_;
      cfg.frame_rate = param->sfsort_param_.frame_rate_;
      cfg.max_obs = param->sfsort_param_.max_obs_;
      tracker_ = new sfsort::SFSORTTracker(cfg);
      break;
    }
    case kTrackerTypeOccluBoost: {
      occluboost::Config cfg;
      cfg.max_age = param->occluboost_param_.max_age_;
      cfg.min_hits = param->occluboost_param_.min_hits_;
      cfg.det_thresh = param->occluboost_param_.det_thresh_;
      cfg.iou_threshold = param->occluboost_param_.iou_threshold_;
      cfg.min_box_area = param->occluboost_param_.min_box_area_;
      cfg.aspect_ratio_thresh = param->occluboost_param_.aspect_ratio_thresh_;
      cfg.lambda_iou = param->occluboost_param_.lambda_iou_;
      cfg.lambda_mhd = param->occluboost_param_.lambda_mhd_;
      cfg.lambda_shape = param->occluboost_param_.lambda_shape_;
      cfg.use_dlo_boost = param->occluboost_param_.use_dlo_boost_;
      cfg.use_duo_boost = param->occluboost_param_.use_duo_boost_;
      cfg.dlo_boost_coef = param->occluboost_param_.dlo_boost_coef_;
      cfg.s_sim_corr = param->occluboost_param_.s_sim_corr_;
      cfg.use_rich_s = param->occluboost_param_.use_rich_s_;
      cfg.use_sb = param->occluboost_param_.use_sb_;
      cfg.use_vt = param->occluboost_param_.use_vt_;
      cfg.with_reid = param->occluboost_param_.with_reid_;
      cfg.cmc_method = param->occluboost_param_.cmc_method_;
      cfg.max_obs = param->occluboost_param_.max_obs_;
      cfg.recovery_appearance_thresh =
          param->occluboost_param_.recovery_appearance_thresh_;
      cfg.recovery_iou_thresh = param->occluboost_param_.recovery_iou_thresh_;
      cfg.recovery_max_age = param->occluboost_param_.recovery_max_age_;
      cfg.feat_alpha = param->occluboost_param_.feat_alpha_;
      cfg.track_low_thresh = param->occluboost_param_.track_low_thresh_;
      cfg.second_iou_thresh = param->occluboost_param_.second_iou_thresh_;
      cfg.second_appearance_thresh =
          param->occluboost_param_.second_appearance_thresh_;
      cfg.second_pass_max_age = param->occluboost_param_.second_pass_max_age_;
      cfg.second_pass_min_hits = param->occluboost_param_.second_pass_min_hits_;
      cfg.use_second_pass = param->occluboost_param_.use_second_pass_;
      cfg.new_track_thresh = param->occluboost_param_.new_track_thresh_;
      cfg.confirm_hits = param->occluboost_param_.confirm_hits_;
      cfg.instant_confirm_thresh =
          param->occluboost_param_.instant_confirm_thresh_;
      cfg.tentative_max_age = param->occluboost_param_.tentative_max_age_;
      cfg.duplicate_iou_thresh = param->occluboost_param_.duplicate_iou_thresh_;
      cfg.ams_enabled = param->occluboost_param_.ams_enabled_;
      cfg.ams_alpha0 = param->occluboost_param_.ams_alpha0_;
      cfg.ams_threshold = param->occluboost_param_.ams_threshold_;
      cfg.ams_buffer_size = param->occluboost_param_.ams_buffer_size_;
      cfg.ams_shrink_ratio = param->occluboost_param_.ams_shrink_ratio_;
      cfg.lambda_emb_multiplier =
          param->occluboost_param_.lambda_emb_multiplier_;
      cfg.reid_preprocess = param->occluboost_param_.reid_preprocess_;
      cfg.reid_device = param->occluboost_param_.reid_device_;
      if (!param->reid_param_.reid_model_path_.empty()) {
        cfg.reid_model_path = param->reid_param_.reid_model_path_;
      }
      tracker_ = new occluboost::OccluBoostTracker(cfg);
      break;
    }
    default:
      NNDEPLOY_LOGE("BoxMotNode: unknown tracker_type_=%d\n",
                    static_cast<int>(active_tracker_type_));
      return base::kStatusCodeErrorInvalidValue;
  }

  return base::kStatusCodeOk;
}

// =========================================================================
// deinit: destroy tracker
// =========================================================================

base::Status BoxMotNode::deinit() {
  if (tracker_) {
    switch (active_tracker_type_) {
      case kTrackerTypeByteTrack:
        delete static_cast<bytetrack::ByteTrackTracker*>(tracker_);
        break;
      case kTrackerTypeBotSort:
        delete static_cast<botsort::BotSortTracker*>(tracker_);
        break;
      case kTrackerTypeOcSort:
        delete static_cast<ocsort::OCSORTTracker*>(tracker_);
        break;
      case kTrackerTypeSfSort:
        delete static_cast<sfsort::SFSORTTracker*>(tracker_);
        break;
      case kTrackerTypeOccluBoost:
        delete static_cast<occluboost::OccluBoostTracker*>(tracker_);
        break;
      default:
        break;
    }
    tracker_ = nullptr;
  }
  return base::kStatusCodeOk;
}

// =========================================================================
// processTracker: templated tracking dispatch
// =========================================================================

template <typename TrackerT, typename DetectionT, typename TrackOutputT>
base::Status BoxMotNode::processTracker(const cv::Mat& frame) {
  auto* tracker = static_cast<TrackerT*>(tracker_);

  // input_[0]/[1] 必须来自不同的边，否则 static_cast 会导致 UB
  detect::BBoxResult* bbox_result =
      static_cast<detect::BBoxResult*>(inputs_[0]->getParam(this));
  detect::ObbResult* obb_result = nullptr;
  if (inputs_.size() > 1 && inputs_[1] != inputs_[0]) {
    obb_result = static_cast<detect::ObbResult*>(inputs_[1]->getParam(this));
  }

  std::vector<DetectionT> detections;
  if (bbox_result && !bbox_result->bboxs_.empty()) {
    detections = bboxResultToDetections<DetectionT>(bbox_result);
  } else if (obb_result && !obb_result->boxes_.empty()) {
    detections = obbResultToDetections<DetectionT>(obb_result);
  } else {
    MOTResult* mot_result = new MOTResult();
    BoxMotResult* boxmot_result = new BoxMotResult();
    outputs_[0]->set(mot_result, false);
    outputs_[1]->set(boxmot_result, false);
    return base::kStatusCodeOk;
  }

  // Run tracking (capital U!)
  std::vector<TrackOutputT> track_outputs = tracker->Update(detections, frame);

  // Convert to nndeploy output types
  MOTResult* mot_result =
      new MOTResult(trackOutputToMOTResult<TrackOutputT>(track_outputs));
  BoxMotResult* boxmot_result =
      new BoxMotResult(trackOutputToBoxMotResult<TrackOutputT>(track_outputs));

  outputs_[0]->set(mot_result, false);
  outputs_[1]->set(boxmot_result, false);
  return base::kStatusCodeOk;
}

// =========================================================================
// run: dispatch to correct tracker type
// =========================================================================

base::Status BoxMotNode::run() {
  // Get frame image from input[2]
  cv::Mat* frame = inputs_[2]->getCvMat(this);
  if (frame == nullptr || frame->empty()) {
    NNDEPLOY_LOGE("BoxMotNode: input image is null or empty\n");
    return base::kStatusCodeErrorInvalidValue;
  }

  switch (active_tracker_type_) {
    case kTrackerTypeByteTrack:
      return processTracker<bytetrack::ByteTrackTracker, bytetrack::Detection,
                            bytetrack::TrackOutput>(*frame);
    case kTrackerTypeBotSort:
      return processTracker<botsort::BotSortTracker, botsort::Detection,
                            botsort::TrackOutput>(*frame);
    case kTrackerTypeOcSort:
      return processTracker<ocsort::OCSORTTracker, ocsort::Detection,
                            ocsort::TrackOutput>(*frame);
    case kTrackerTypeSfSort:
      return processTracker<sfsort::SFSORTTracker, sfsort::Detection,
                            sfsort::TrackOutput>(*frame);
    case kTrackerTypeOccluBoost:
      return processTracker<occluboost::OccluBoostTracker,
                            occluboost::Detection, occluboost::TrackOutput>(
          *frame);
    default:
      NNDEPLOY_LOGE("BoxMotNode: unknown tracker_type_=%d\n",
                    static_cast<int>(active_tracker_type_));
      return base::kStatusCodeErrorInvalidValue;
  }
}

REGISTER_NODE("nndeploy::track::boxmot::BoxMotNode", BoxMotNode);

}  // namespace track
}  // namespace nndeploy
