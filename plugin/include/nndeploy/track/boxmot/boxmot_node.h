#ifndef _NNDEPLOY_TRACK_BOXMOT_BOXMOT_NODE_H_
#define _NNDEPLOY_TRACK_BOXMOT_BOXMOT_NODE_H_

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
#include "nndeploy/dag/edge.h"
#include "nndeploy/dag/node.h"
#include "nndeploy/detect/result.h"
#include "nndeploy/detect/yolo_obb/result.h"
#include "nndeploy/device/buffer.h"
#include "nndeploy/device/device.h"
#include "nndeploy/device/memory_pool.h"
#include "nndeploy/device/tensor.h"
#include "nndeploy/track/boxmot/result.h"
#include "nndeploy/track/result.h"

namespace nndeploy {
namespace track {

/**
 * @brief Unified BoxMot tracking node
 *
 * Single node class that wraps all 5 BoxMot native C++ trackers.
 * The tracker_type_ field in BoxMotParam selects which tracker is used.
 *
 * Inputs:
 *   [0] detect::BBoxResult        (AABB detections, required)
 *   [1] detect::ObbResult         (OBB detections, optional)
 *   [2] cv::Mat                   (frame image, required)
 *
 * Outputs:
 *   [0] MOTResult           (legacy-compatible tracked objects)
 *   [1] BoxMotResult        (extended tracking result with OBB info)
 *
 * Registration key: "nndeploy::track::boxmot::BoxMotNode"
 */
class NNDEPLOY_CC_API BoxMotNode : public dag::Node {
 public:
  BoxMotNode(const std::string& name) : Node(name) {
    key_ = "nndeploy::track::boxmot::BoxMotNode";
    desc_ =
        "BoxMot unified multi-object tracker "
        "[BBoxResult/ObbResult+Image->MOTResult+BoxMotResult]";
    param_ = std::make_shared<BoxMotParam>();
    this->setInputTypeInfo<detect::BBoxResult>();
    this->setInputTypeInfo<detect::ObbResult>();
    this->setInputTypeInfo<cv::Mat>();
    this->setOutputTypeInfo<MOTResult>();
    this->setOutputTypeInfo<BoxMotResult>();
  }
  BoxMotNode(const std::string& name, std::vector<dag::Edge*> inputs,
             std::vector<dag::Edge*> outputs)
      : Node(name, inputs, outputs) {
    key_ = "nndeploy::track::boxmot::BoxMotNode";
    desc_ =
        "BoxMot unified multi-object tracker "
        "[BBoxResult/ObbResult+Image->MOTResult+BoxMotResult]";
    param_ = std::make_shared<BoxMotParam>();
    this->setInputTypeInfo<detect::BBoxResult>();
    this->setInputTypeInfo<detect::ObbResult>();
    this->setInputTypeInfo<cv::Mat>();
    this->setOutputTypeInfo<MOTResult>();
    this->setOutputTypeInfo<BoxMotResult>();
  }
  virtual ~BoxMotNode() {}

  virtual base::Status init();
  virtual base::Status deinit();
  virtual base::Status run();

 private:
  /**
   * @brief Process tracking for a specific tracker type.
   *
   * Templated to handle per-tracker Detection/TrackOutput types.
   * Defined in boxmot_node.cc via explicit instantiation.
   */
  template <typename TrackerT, typename DetectionT, typename TrackOutputT>
  base::Status processTracker(const cv::Mat& frame);

  // Opaque pointer to avoid including tracker headers in the public header.
  // Actual type is one of: bytetrack::ByteTrackTracker*,
  // botsort::BotSortTracker*, ocsort::OCSORTTracker*, sfsort::SFSORTTracker*,
  // occluboost::OccluBoostTracker*
  void* tracker_ = nullptr;
  TrackerType active_tracker_type_ = kTrackerTypeByteTrack;
};

}  // namespace track
}  // namespace nndeploy

#endif  // _NNDEPLOY_TRACK_BOXMOT_BOXMOT_NODE_H_
