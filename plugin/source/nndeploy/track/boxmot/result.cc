#include "nndeploy/track/boxmot/result.h"

namespace nndeploy {
namespace track {

// ---------------------------------------------------------------------------
// ReIDParam
// ---------------------------------------------------------------------------

base::Status ReIDParam::serialize(
    rapidjson::Value& json, rapidjson::Document::AllocatorType& allocator) {
  json.AddMember("reid_model_path_",
                 rapidjson::Value(reid_model_path_.c_str(), allocator),
                 allocator);
  json.AddMember("reid_backend_", reid_backend_, allocator);
  json.AddMember("use_external_embedding_", use_external_embedding_, allocator);
  return base::kStatusCodeOk;
}

base::Status ReIDParam::deserialize(rapidjson::Value& json) {
  if (json.HasMember("reid_model_path_"))
    reid_model_path_ = json["reid_model_path_"].GetString();
  if (json.HasMember("reid_backend_"))
    reid_backend_ = json["reid_backend_"].GetInt();
  if (json.HasMember("use_external_embedding_"))
    use_external_embedding_ = json["use_external_embedding_"].GetBool();
  return base::kStatusCodeOk;
}

// ---------------------------------------------------------------------------
// ByteTrackBoxMotParam
// ---------------------------------------------------------------------------

base::Status ByteTrackBoxMotParam::serialize(
    rapidjson::Value& json, rapidjson::Document::AllocatorType& allocator) {
  json.AddMember("min_conf_", min_conf_, allocator);
  json.AddMember("track_thresh_", track_thresh_, allocator);
  json.AddMember("match_thresh_", match_thresh_, allocator);
  json.AddMember("track_buffer_", track_buffer_, allocator);
  json.AddMember("frame_rate_", frame_rate_, allocator);
  json.AddMember("max_obs_", max_obs_, allocator);
  return base::kStatusCodeOk;
}

base::Status ByteTrackBoxMotParam::deserialize(rapidjson::Value& json) {
  if (json.HasMember("min_conf_")) min_conf_ = json["min_conf_"].GetFloat();
  if (json.HasMember("track_thresh_"))
    track_thresh_ = json["track_thresh_"].GetFloat();
  if (json.HasMember("match_thresh_"))
    match_thresh_ = json["match_thresh_"].GetFloat();
  if (json.HasMember("track_buffer_"))
    track_buffer_ = json["track_buffer_"].GetInt();
  if (json.HasMember("frame_rate_")) frame_rate_ = json["frame_rate_"].GetInt();
  if (json.HasMember("max_obs_")) max_obs_ = json["max_obs_"].GetInt();
  return base::kStatusCodeOk;
}

// ---------------------------------------------------------------------------
// BotSortBoxMotParam
// ---------------------------------------------------------------------------

base::Status BotSortBoxMotParam::serialize(
    rapidjson::Value& json, rapidjson::Document::AllocatorType& allocator) {
  json.AddMember("track_high_thresh_", track_high_thresh_, allocator);
  json.AddMember("track_low_thresh_", track_low_thresh_, allocator);
  json.AddMember("new_track_thresh_", new_track_thresh_, allocator);
  json.AddMember("track_buffer_", track_buffer_, allocator);
  json.AddMember("match_thresh_", match_thresh_, allocator);
  json.AddMember("proximity_thresh_", proximity_thresh_, allocator);
  json.AddMember("appearance_thresh_", appearance_thresh_, allocator);
  json.AddMember("cmc_method_",
                 rapidjson::Value(cmc_method_.c_str(), allocator), allocator);
  json.AddMember("frame_rate_", frame_rate_, allocator);
  json.AddMember("fuse_first_associate_", fuse_first_associate_, allocator);
  json.AddMember("with_reid_", with_reid_, allocator);
  json.AddMember("max_obs_", max_obs_, allocator);
  json.AddMember("reid_preprocess_",
                 rapidjson::Value(reid_preprocess_.c_str(), allocator),
                 allocator);
  return base::kStatusCodeOk;
}

base::Status BotSortBoxMotParam::deserialize(rapidjson::Value& json) {
  if (json.HasMember("track_high_thresh_"))
    track_high_thresh_ = json["track_high_thresh_"].GetFloat();
  if (json.HasMember("track_low_thresh_"))
    track_low_thresh_ = json["track_low_thresh_"].GetFloat();
  if (json.HasMember("new_track_thresh_"))
    new_track_thresh_ = json["new_track_thresh_"].GetFloat();
  if (json.HasMember("track_buffer_"))
    track_buffer_ = json["track_buffer_"].GetInt();
  if (json.HasMember("match_thresh_"))
    match_thresh_ = json["match_thresh_"].GetFloat();
  if (json.HasMember("proximity_thresh_"))
    proximity_thresh_ = json["proximity_thresh_"].GetFloat();
  if (json.HasMember("appearance_thresh_"))
    appearance_thresh_ = json["appearance_thresh_"].GetFloat();
  if (json.HasMember("cmc_method_"))
    cmc_method_ = json["cmc_method_"].GetString();
  if (json.HasMember("frame_rate_")) frame_rate_ = json["frame_rate_"].GetInt();
  if (json.HasMember("fuse_first_associate_"))
    fuse_first_associate_ = json["fuse_first_associate_"].GetBool();
  if (json.HasMember("with_reid_")) with_reid_ = json["with_reid_"].GetBool();
  if (json.HasMember("max_obs_")) max_obs_ = json["max_obs_"].GetInt();
  if (json.HasMember("reid_preprocess_"))
    reid_preprocess_ = json["reid_preprocess_"].GetString();
  return base::kStatusCodeOk;
}

// ---------------------------------------------------------------------------
// OcSortBoxMotParam
// ---------------------------------------------------------------------------

base::Status OcSortBoxMotParam::serialize(
    rapidjson::Value& json, rapidjson::Document::AllocatorType& allocator) {
  json.AddMember("min_conf_", min_conf_, allocator);
  json.AddMember("det_thresh_", det_thresh_, allocator);
  json.AddMember("iou_threshold_", iou_threshold_, allocator);
  json.AddMember("max_age_", max_age_, allocator);
  json.AddMember("min_hits_", min_hits_, allocator);
  json.AddMember("delta_t_", delta_t_, allocator);
  json.AddMember("use_byte_", use_byte_, allocator);
  json.AddMember("inertia_", inertia_, allocator);
  json.AddMember("q_xy_scaling_", q_xy_scaling_, allocator);
  json.AddMember("q_s_scaling_", q_s_scaling_, allocator);
  json.AddMember("max_obs_", max_obs_, allocator);
  return base::kStatusCodeOk;
}

base::Status OcSortBoxMotParam::deserialize(rapidjson::Value& json) {
  if (json.HasMember("min_conf_")) min_conf_ = json["min_conf_"].GetFloat();
  if (json.HasMember("det_thresh_"))
    det_thresh_ = json["det_thresh_"].GetFloat();
  if (json.HasMember("iou_threshold_"))
    iou_threshold_ = json["iou_threshold_"].GetFloat();
  if (json.HasMember("max_age_")) max_age_ = json["max_age_"].GetInt();
  if (json.HasMember("min_hits_")) min_hits_ = json["min_hits_"].GetInt();
  if (json.HasMember("delta_t_")) delta_t_ = json["delta_t_"].GetInt();
  if (json.HasMember("use_byte_")) use_byte_ = json["use_byte_"].GetBool();
  if (json.HasMember("inertia_")) inertia_ = json["inertia_"].GetFloat();
  if (json.HasMember("q_xy_scaling_"))
    q_xy_scaling_ = json["q_xy_scaling_"].GetFloat();
  if (json.HasMember("q_s_scaling_"))
    q_s_scaling_ = json["q_s_scaling_"].GetFloat();
  if (json.HasMember("max_obs_")) max_obs_ = json["max_obs_"].GetInt();
  return base::kStatusCodeOk;
}

// ---------------------------------------------------------------------------
// SfSortBoxMotParam
// ---------------------------------------------------------------------------

base::Status SfSortBoxMotParam::serialize(
    rapidjson::Value& json, rapidjson::Document::AllocatorType& allocator) {
  json.AddMember("high_th_", high_th_, allocator);
  json.AddMember("match_th_first_", match_th_first_, allocator);
  json.AddMember("new_track_th_", new_track_th_, allocator);
  json.AddMember("low_th_", low_th_, allocator);
  json.AddMember("match_th_second_", match_th_second_, allocator);
  json.AddMember("dynamic_tuning_", dynamic_tuning_, allocator);
  json.AddMember("cth_", cth_, allocator);
  json.AddMember("high_th_m_", high_th_m_, allocator);
  json.AddMember("new_track_th_m_", new_track_th_m_, allocator);
  json.AddMember("match_th_first_m_", match_th_first_m_, allocator);
  json.AddMember("obb_theta_damping_", obb_theta_damping_, allocator);
  json.AddMember("marginal_timeout_", marginal_timeout_, allocator);
  json.AddMember("central_timeout_", central_timeout_, allocator);
  json.AddMember("frame_width_", frame_width_, allocator);
  json.AddMember("frame_height_", frame_height_, allocator);
  json.AddMember("horizontal_margin_", horizontal_margin_, allocator);
  json.AddMember("vertical_margin_", vertical_margin_, allocator);
  json.AddMember("frame_rate_", frame_rate_, allocator);
  json.AddMember("max_obs_", max_obs_, allocator);
  return base::kStatusCodeOk;
}

base::Status SfSortBoxMotParam::deserialize(rapidjson::Value& json) {
  if (json.HasMember("high_th_")) high_th_ = json["high_th_"].GetFloat();
  if (json.HasMember("match_th_first_"))
    match_th_first_ = json["match_th_first_"].GetFloat();
  if (json.HasMember("new_track_th_"))
    new_track_th_ = json["new_track_th_"].GetFloat();
  if (json.HasMember("low_th_")) low_th_ = json["low_th_"].GetFloat();
  if (json.HasMember("match_th_second_"))
    match_th_second_ = json["match_th_second_"].GetFloat();
  if (json.HasMember("dynamic_tuning_"))
    dynamic_tuning_ = json["dynamic_tuning_"].GetBool();
  if (json.HasMember("cth_")) cth_ = json["cth_"].GetFloat();
  if (json.HasMember("high_th_m_")) high_th_m_ = json["high_th_m_"].GetFloat();
  if (json.HasMember("new_track_th_m_"))
    new_track_th_m_ = json["new_track_th_m_"].GetFloat();
  if (json.HasMember("match_th_first_m_"))
    match_th_first_m_ = json["match_th_first_m_"].GetFloat();
  if (json.HasMember("obb_theta_damping_"))
    obb_theta_damping_ = json["obb_theta_damping_"].GetFloat();
  if (json.HasMember("marginal_timeout_"))
    marginal_timeout_ = json["marginal_timeout_"].GetInt();
  if (json.HasMember("central_timeout_"))
    central_timeout_ = json["central_timeout_"].GetInt();
  if (json.HasMember("frame_width_"))
    frame_width_ = json["frame_width_"].GetInt();
  if (json.HasMember("frame_height_"))
    frame_height_ = json["frame_height_"].GetInt();
  if (json.HasMember("horizontal_margin_"))
    horizontal_margin_ = json["horizontal_margin_"].GetInt();
  if (json.HasMember("vertical_margin_"))
    vertical_margin_ = json["vertical_margin_"].GetInt();
  if (json.HasMember("frame_rate_")) frame_rate_ = json["frame_rate_"].GetInt();
  if (json.HasMember("max_obs_")) max_obs_ = json["max_obs_"].GetInt();
  return base::kStatusCodeOk;
}

// ---------------------------------------------------------------------------
// OccluBoostBoxMotParam
// ---------------------------------------------------------------------------

base::Status OccluBoostBoxMotParam::serialize(
    rapidjson::Value& json, rapidjson::Document::AllocatorType& allocator) {
  // BoostTrack inherited
  json.AddMember("max_age_", max_age_, allocator);
  json.AddMember("min_hits_", min_hits_, allocator);
  json.AddMember("det_thresh_", det_thresh_, allocator);
  json.AddMember("iou_threshold_", iou_threshold_, allocator);
  json.AddMember("min_box_area_", min_box_area_, allocator);
  json.AddMember("aspect_ratio_thresh_", aspect_ratio_thresh_, allocator);
  json.AddMember("lambda_iou_", lambda_iou_, allocator);
  json.AddMember("lambda_mhd_", lambda_mhd_, allocator);
  json.AddMember("lambda_shape_", lambda_shape_, allocator);
  json.AddMember("use_dlo_boost_", use_dlo_boost_, allocator);
  json.AddMember("use_duo_boost_", use_duo_boost_, allocator);
  json.AddMember("dlo_boost_coef_", dlo_boost_coef_, allocator);
  json.AddMember("s_sim_corr_", s_sim_corr_, allocator);
  json.AddMember("use_rich_s_", use_rich_s_, allocator);
  json.AddMember("use_sb_", use_sb_, allocator);
  json.AddMember("use_vt_", use_vt_, allocator);
  json.AddMember("with_reid_", with_reid_, allocator);
  json.AddMember("cmc_method_",
                 rapidjson::Value(cmc_method_.c_str(), allocator), allocator);
  json.AddMember("max_obs_", max_obs_, allocator);
  // OccluBoost specific
  json.AddMember("recovery_appearance_thresh_", recovery_appearance_thresh_,
                 allocator);
  json.AddMember("recovery_iou_thresh_", recovery_iou_thresh_, allocator);
  json.AddMember("recovery_max_age_", recovery_max_age_, allocator);
  json.AddMember("feat_alpha_", feat_alpha_, allocator);
  json.AddMember("track_low_thresh_", track_low_thresh_, allocator);
  json.AddMember("second_iou_thresh_", second_iou_thresh_, allocator);
  json.AddMember("second_appearance_thresh_", second_appearance_thresh_,
                 allocator);
  json.AddMember("second_pass_max_age_", second_pass_max_age_, allocator);
  json.AddMember("second_pass_min_hits_", second_pass_min_hits_, allocator);
  json.AddMember("use_second_pass_", use_second_pass_, allocator);
  json.AddMember("new_track_thresh_", new_track_thresh_, allocator);
  json.AddMember("confirm_hits_", confirm_hits_, allocator);
  json.AddMember("instant_confirm_thresh_", instant_confirm_thresh_, allocator);
  json.AddMember("tentative_max_age_", tentative_max_age_, allocator);
  json.AddMember("duplicate_iou_thresh_", duplicate_iou_thresh_, allocator);
  json.AddMember("ams_enabled_", ams_enabled_, allocator);
  json.AddMember("ams_alpha0_", ams_alpha0_, allocator);
  json.AddMember("ams_threshold_", ams_threshold_, allocator);
  json.AddMember("ams_buffer_size_", ams_buffer_size_, allocator);
  json.AddMember("ams_shrink_ratio_", ams_shrink_ratio_, allocator);
  json.AddMember("lambda_emb_multiplier_", lambda_emb_multiplier_, allocator);
  // ReID
  json.AddMember("reid_preprocess_",
                 rapidjson::Value(reid_preprocess_.c_str(), allocator),
                 allocator);
  json.AddMember("reid_device_",
                 rapidjson::Value(reid_device_.c_str(), allocator), allocator);
  return base::kStatusCodeOk;
}

base::Status OccluBoostBoxMotParam::deserialize(rapidjson::Value& json) {
  if (json.HasMember("max_age_")) max_age_ = json["max_age_"].GetInt();
  if (json.HasMember("min_hits_")) min_hits_ = json["min_hits_"].GetInt();
  if (json.HasMember("det_thresh_"))
    det_thresh_ = json["det_thresh_"].GetFloat();
  if (json.HasMember("iou_threshold_"))
    iou_threshold_ = json["iou_threshold_"].GetFloat();
  if (json.HasMember("min_box_area_"))
    min_box_area_ = json["min_box_area_"].GetInt();
  if (json.HasMember("aspect_ratio_thresh_"))
    aspect_ratio_thresh_ = json["aspect_ratio_thresh_"].GetFloat();
  if (json.HasMember("lambda_iou_"))
    lambda_iou_ = json["lambda_iou_"].GetFloat();
  if (json.HasMember("lambda_mhd_"))
    lambda_mhd_ = json["lambda_mhd_"].GetFloat();
  if (json.HasMember("lambda_shape_"))
    lambda_shape_ = json["lambda_shape_"].GetFloat();
  if (json.HasMember("use_dlo_boost_"))
    use_dlo_boost_ = json["use_dlo_boost_"].GetBool();
  if (json.HasMember("use_duo_boost_"))
    use_duo_boost_ = json["use_duo_boost_"].GetBool();
  if (json.HasMember("dlo_boost_coef_"))
    dlo_boost_coef_ = json["dlo_boost_coef_"].GetFloat();
  if (json.HasMember("s_sim_corr_"))
    s_sim_corr_ = json["s_sim_corr_"].GetBool();
  if (json.HasMember("use_rich_s_"))
    use_rich_s_ = json["use_rich_s_"].GetBool();
  if (json.HasMember("use_sb_")) use_sb_ = json["use_sb_"].GetBool();
  if (json.HasMember("use_vt_")) use_vt_ = json["use_vt_"].GetBool();
  if (json.HasMember("with_reid_")) with_reid_ = json["with_reid_"].GetBool();
  if (json.HasMember("cmc_method_"))
    cmc_method_ = json["cmc_method_"].GetString();
  if (json.HasMember("max_obs_")) max_obs_ = json["max_obs_"].GetInt();
  if (json.HasMember("recovery_appearance_thresh_"))
    recovery_appearance_thresh_ =
        json["recovery_appearance_thresh_"].GetFloat();
  if (json.HasMember("recovery_iou_thresh_"))
    recovery_iou_thresh_ = json["recovery_iou_thresh_"].GetFloat();
  if (json.HasMember("recovery_max_age_"))
    recovery_max_age_ = json["recovery_max_age_"].GetInt();
  if (json.HasMember("feat_alpha_"))
    feat_alpha_ = json["feat_alpha_"].GetFloat();
  if (json.HasMember("track_low_thresh_"))
    track_low_thresh_ = json["track_low_thresh_"].GetFloat();
  if (json.HasMember("second_iou_thresh_"))
    second_iou_thresh_ = json["second_iou_thresh_"].GetFloat();
  if (json.HasMember("second_appearance_thresh_"))
    second_appearance_thresh_ = json["second_appearance_thresh_"].GetFloat();
  if (json.HasMember("second_pass_max_age_"))
    second_pass_max_age_ = json["second_pass_max_age_"].GetInt();
  if (json.HasMember("second_pass_min_hits_"))
    second_pass_min_hits_ = json["second_pass_min_hits_"].GetInt();
  if (json.HasMember("use_second_pass_"))
    use_second_pass_ = json["use_second_pass_"].GetBool();
  if (json.HasMember("new_track_thresh_"))
    new_track_thresh_ = json["new_track_thresh_"].GetFloat();
  if (json.HasMember("confirm_hits_"))
    confirm_hits_ = json["confirm_hits_"].GetInt();
  if (json.HasMember("instant_confirm_thresh_"))
    instant_confirm_thresh_ = json["instant_confirm_thresh_"].GetFloat();
  if (json.HasMember("tentative_max_age_"))
    tentative_max_age_ = json["tentative_max_age_"].GetInt();
  if (json.HasMember("duplicate_iou_thresh_"))
    duplicate_iou_thresh_ = json["duplicate_iou_thresh_"].GetFloat();
  if (json.HasMember("ams_enabled_"))
    ams_enabled_ = json["ams_enabled_"].GetBool();
  if (json.HasMember("ams_alpha0_"))
    ams_alpha0_ = json["ams_alpha0_"].GetFloat();
  if (json.HasMember("ams_threshold_"))
    ams_threshold_ = json["ams_threshold_"].GetFloat();
  if (json.HasMember("ams_buffer_size_"))
    ams_buffer_size_ = json["ams_buffer_size_"].GetInt();
  if (json.HasMember("ams_shrink_ratio_"))
    ams_shrink_ratio_ = json["ams_shrink_ratio_"].GetFloat();
  if (json.HasMember("lambda_emb_multiplier_"))
    lambda_emb_multiplier_ = json["lambda_emb_multiplier_"].GetFloat();
  if (json.HasMember("reid_preprocess_"))
    reid_preprocess_ = json["reid_preprocess_"].GetString();
  if (json.HasMember("reid_device_"))
    reid_device_ = json["reid_device_"].GetString();
  return base::kStatusCodeOk;
}

// ---------------------------------------------------------------------------
// BoxMotParam
// ---------------------------------------------------------------------------

base::Status BoxMotParam::serialize(
    rapidjson::Value& json, rapidjson::Document::AllocatorType& allocator) {
  json.AddMember("tracker_type_", static_cast<int>(tracker_type_), allocator);

  // ReID param
  {
    rapidjson::Value reid_json(rapidjson::kObjectType);
    reid_param_.serialize(reid_json, allocator);
    json.AddMember("reid_param_", reid_json, allocator);
  }
  // ByteTrack param
  {
    rapidjson::Value bt_json(rapidjson::kObjectType);
    bytetrack_param_.serialize(bt_json, allocator);
    json.AddMember("bytetrack_param_", bt_json, allocator);
  }
  // BotSort param
  {
    rapidjson::Value bs_json(rapidjson::kObjectType);
    botsort_param_.serialize(bs_json, allocator);
    json.AddMember("botsort_param_", bs_json, allocator);
  }
  // OcSort param
  {
    rapidjson::Value os_json(rapidjson::kObjectType);
    ocsort_param_.serialize(os_json, allocator);
    json.AddMember("ocsort_param_", os_json, allocator);
  }
  // SfSort param
  {
    rapidjson::Value ss_json(rapidjson::kObjectType);
    sfsort_param_.serialize(ss_json, allocator);
    json.AddMember("sfsort_param_", ss_json, allocator);
  }
  // OccluBoost param
  {
    rapidjson::Value ob_json(rapidjson::kObjectType);
    occluboost_param_.serialize(ob_json, allocator);
    json.AddMember("occluboost_param_", ob_json, allocator);
  }
  return base::kStatusCodeOk;
}

base::Status BoxMotParam::deserialize(rapidjson::Value& json) {
  if (json.HasMember("tracker_type_"))
    tracker_type_ = static_cast<TrackerType>(json["tracker_type_"].GetInt());
  if (json.HasMember("reid_param_"))
    reid_param_.deserialize(json["reid_param_"]);
  if (json.HasMember("bytetrack_param_"))
    bytetrack_param_.deserialize(json["bytetrack_param_"]);
  if (json.HasMember("botsort_param_"))
    botsort_param_.deserialize(json["botsort_param_"]);
  if (json.HasMember("ocsort_param_"))
    ocsort_param_.deserialize(json["ocsort_param_"]);
  if (json.HasMember("sfsort_param_"))
    sfsort_param_.deserialize(json["sfsort_param_"]);
  if (json.HasMember("occluboost_param_"))
    occluboost_param_.deserialize(json["occluboost_param_"]);
  return base::kStatusCodeOk;
}

}  // namespace track
}  // namespace nndeploy
