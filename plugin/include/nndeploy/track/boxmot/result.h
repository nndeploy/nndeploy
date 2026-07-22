#ifndef _NNDEPLOY_TRACK_BOXMOT_RESULT_H_
#define _NNDEPLOY_TRACK_BOXMOT_RESULT_H_

#include "nndeploy/base/any.h"
#include "nndeploy/base/common.h"
#include "nndeploy/base/glic_stl_include.h"
#include "nndeploy/base/log.h"
#include "nndeploy/base/macro.h"
#include "nndeploy/base/object.h"
#include "nndeploy/base/param.h"
#include "nndeploy/base/status.h"
#include "nndeploy/base/string.h"
#include "nndeploy/base/type.h"
#include "nndeploy/device/buffer.h"
#include "nndeploy/device/device.h"
#include "nndeploy/device/memory_pool.h"
#include "nndeploy/device/tensor.h"

namespace nndeploy {
namespace track {

/**
 * @brief BoxMot tracker type enumeration
 */
enum TrackerType : int {
  kTrackerTypeByteTrack = 0,
  kTrackerTypeBotSort = 1,
  kTrackerTypeOcSort = 2,
  kTrackerTypeSfSort = 3,
  kTrackerTypeOccluBoost = 4,
};

/**
 * @brief Single tracked object from BoxMot
 *
 * Supports both AABB (bbox) and OBB (obb) dual mode.
 * When is_obb is false, bbox_ contains [xmin, ymin, xmax, ymax].
 * When is_obb is true, obb_ contains [cx, cy, w, h, angle_rad].
 */
struct NNDEPLOY_CC_API BoxMotTrack {
  int id_ = -1;
  std::array<float, 4> bbox_ = {0, 0, 0, 0};
  std::array<float, 5> obb_ = {0, 0, 0, 0, 0};
  float confidence_ = 0.0f;
  int class_id_ = 0;
  int detection_index_ = -1;
  bool is_obb_ = false;
};

/**
 * @brief BoxMot tracking result for a single frame
 */
class NNDEPLOY_CC_API BoxMotResult : public base::Param {
 public:
  BoxMotResult() {}
  virtual ~BoxMotResult() {}

  std::vector<BoxMotTrack> tracks_;
  int frame_id_ = 0;
};

// ---------------------------------------------------------------------------
// Per-tracker parameter structs
// ---------------------------------------------------------------------------

/**
 * @brief ReID configuration
 */
class NNDEPLOY_CC_API ReIDParam : public base::Param {
 public:
  std::string reid_model_path_;
  int reid_backend_ = 0;  // base::InferenceType value
  bool use_external_embedding_ = false;

  virtual base::Status serialize(rapidjson::Value& json,
                                 rapidjson::Document::AllocatorType& allocator);
  virtual base::Status deserialize(rapidjson::Value& json);
};

/**
 * @brief ByteTrack tracker parameters
 */
class NNDEPLOY_CC_API ByteTrackBoxMotParam : public base::Param {
 public:
  float min_conf_ = 0.1f;
  float track_thresh_ = 0.6f;
  float match_thresh_ = 0.9f;
  int track_buffer_ = 30;
  int frame_rate_ = 30;
  int max_obs_ = 50;

  virtual base::Status serialize(rapidjson::Value& json,
                                 rapidjson::Document::AllocatorType& allocator);
  virtual base::Status deserialize(rapidjson::Value& json);
};

/**
 * @brief BotSort tracker parameters
 */
class NNDEPLOY_CC_API BotSortBoxMotParam : public base::Param {
 public:
  float track_high_thresh_ = 0.6f;
  float track_low_thresh_ = 0.1f;
  float new_track_thresh_ = 0.7f;
  int track_buffer_ = 30;
  float match_thresh_ = 0.8f;
  float proximity_thresh_ = 0.5f;
  float appearance_thresh_ = 0.25f;
  std::string cmc_method_ = "ecc";
  int frame_rate_ = 30;
  bool fuse_first_associate_ = false;
  bool with_reid_ = true;
  int max_obs_ = 50;
  std::string reid_preprocess_ = "resize_pad";

  virtual base::Status serialize(rapidjson::Value& json,
                                 rapidjson::Document::AllocatorType& allocator);
  virtual base::Status deserialize(rapidjson::Value& json);
};

/**
 * @brief OcSort tracker parameters
 */
class NNDEPLOY_CC_API OcSortBoxMotParam : public base::Param {
 public:
  float min_conf_ = 0.1f;
  float det_thresh_ = 0.6f;
  float iou_threshold_ = 0.3f;
  int max_age_ = 30;
  int min_hits_ = 3;
  int delta_t_ = 3;
  bool use_byte_ = false;
  float inertia_ = 0.1f;
  float q_xy_scaling_ = 0.01f;
  float q_s_scaling_ = 0.0001f;
  int max_obs_ = 50;

  virtual base::Status serialize(rapidjson::Value& json,
                                 rapidjson::Document::AllocatorType& allocator);
  virtual base::Status deserialize(rapidjson::Value& json);
};

/**
 * @brief SfSort tracker parameters
 */
class NNDEPLOY_CC_API SfSortBoxMotParam : public base::Param {
 public:
  float high_th_ = 0.6f;
  float match_th_first_ = 0.67f;
  float new_track_th_ = 0.7f;
  float low_th_ = 0.1f;
  float match_th_second_ = 0.3f;
  bool dynamic_tuning_ = false;
  float cth_ = 0.5f;
  float high_th_m_ = 0.0f;
  float new_track_th_m_ = 0.0f;
  float match_th_first_m_ = 0.0f;
  float obb_theta_damping_ = 0.8f;
  int marginal_timeout_ = 0;
  int central_timeout_ = 0;
  int frame_width_ = 0;
  int frame_height_ = 0;
  int horizontal_margin_ = 0;
  int vertical_margin_ = 0;
  int frame_rate_ = 30;
  int max_obs_ = 50;

  virtual base::Status serialize(rapidjson::Value& json,
                                 rapidjson::Document::AllocatorType& allocator);
  virtual base::Status deserialize(rapidjson::Value& json);
};

/**
 * @brief OccluBoost tracker parameters
 */
class NNDEPLOY_CC_API OccluBoostBoxMotParam : public base::Param {
 public:
  // BoostTrack inherited
  int max_age_ = 120;
  int min_hits_ = 1;
  float det_thresh_ = 0.6f;
  float iou_threshold_ = 0.2f;
  int min_box_area_ = 10;
  float aspect_ratio_thresh_ = 1.6f;
  float lambda_iou_ = 0.3f;
  float lambda_mhd_ = 0.3f;
  float lambda_shape_ = 0.5f;
  bool use_dlo_boost_ = true;
  bool use_duo_boost_ = false;
  float dlo_boost_coef_ = 0.65f;
  bool s_sim_corr_ = false;
  bool use_rich_s_ = false;
  bool use_sb_ = true;
  bool use_vt_ = false;
  bool with_reid_ = true;
  std::string cmc_method_ = "ecc";
  int max_obs_ = 50;

  // OccluBoost specific
  float recovery_appearance_thresh_ = 0.4f;
  float recovery_iou_thresh_ = 0.2f;
  int recovery_max_age_ = 70;
  float feat_alpha_ = 0.95f;
  float track_low_thresh_ = 0.04f;
  float second_iou_thresh_ = 0.5f;
  float second_appearance_thresh_ = 0.6f;
  int second_pass_max_age_ = 5;
  int second_pass_min_hits_ = 3;
  bool use_second_pass_ = true;
  float new_track_thresh_ = 0.6f;
  int confirm_hits_ = 4;
  float instant_confirm_thresh_ = 0.77f;
  int tentative_max_age_ = 1;
  float duplicate_iou_thresh_ = 0.95f;
  bool ams_enabled_ = true;
  float ams_alpha0_ = 0.4f;
  float ams_threshold_ = 0.5f;
  int ams_buffer_size_ = 30;
  float ams_shrink_ratio_ = 0.75f;
  float lambda_emb_multiplier_ = 1.5f;

  // ReID
  std::string reid_preprocess_ = "resize_pad";
  std::string reid_device_ = "auto";

  virtual base::Status serialize(rapidjson::Value& json,
                                 rapidjson::Document::AllocatorType& allocator);
  virtual base::Status deserialize(rapidjson::Value& json);
};

/**
 * @brief Top-level BoxMot node parameter
 *
 * Contains tracker_type to select which tracker to use,
 * plus sub-parameters for each tracker type.
 */
class NNDEPLOY_CC_API BoxMotParam : public base::Param {
 public:
  TrackerType tracker_type_ = kTrackerTypeByteTrack;
  ReIDParam reid_param_;
  ByteTrackBoxMotParam bytetrack_param_;
  BotSortBoxMotParam botsort_param_;
  OcSortBoxMotParam ocsort_param_;
  SfSortBoxMotParam sfsort_param_;
  OccluBoostBoxMotParam occluboost_param_;

  virtual base::Status serialize(rapidjson::Value& json,
                                 rapidjson::Document::AllocatorType& allocator);
  virtual base::Status deserialize(rapidjson::Value& json);
};

}  // namespace track
}  // namespace nndeploy

#endif  // _NNDEPLOY_TRACK_BOXMOT_RESULT_H_
