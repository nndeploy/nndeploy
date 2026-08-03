#ifndef _NNDEPLOY_TRACK_BOTSORT_BOT_SORT_NODE_H_
#define _NNDEPLOY_TRACK_BOTSORT_BOT_SORT_NODE_H_

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
#include "nndeploy/device/buffer.h"
#include "nndeploy/device/device.h"
#include "nndeploy/device/memory_pool.h"
#include "nndeploy/device/tensor.h"
#include "nndeploy/detect/result.h"
#include "nndeploy/track/botsort/botsort.h"
#include "nndeploy/track/result.h"

namespace nndeploy {
namespace track {

/**
 * @brief Bot-SORT DAG node
 *
 * Input[0]: cv::Mat       (current video frame, for GMC)
 * Input[1]: DetectResult  (from a detection graph/node)
 * Output:   MOTResult     (tracked objects with IDs)
 *
 * Extends ByteTrack with Global Motion Compensation (GMC) using
 * ORB feature matching and camera motion- compensated Kalman filter.
 */
class NNDEPLOY_CC_API BotSortNode : public dag::Node {
 public:
  BotSortNode(const std::string &name) : Node(name) {
    key_ = "nndeploy::track::BotSortNode";
    desc_ = "Bot-SORT tracker with GMC[cv::Mat,BBoxResult->MOTResult]";
    this->setInputTypeInfo<cv::Mat>();
    this->setInputTypeInfo<detect::BBoxResult>();
    this->setOutputTypeInfo<MOTResult>();
  }
  BotSortNode(const std::string &name, std::vector<dag::Edge *> inputs,
              std::vector<dag::Edge *> outputs)
      : Node(name, inputs, outputs) {
    key_ = "nndeploy::track::BotSortNode";
    desc_ = "Bot-SORT tracker with GMC[cv::Mat,BBoxResult->MOTResult]";
    this->setInputTypeInfo<cv::Mat>();
    this->setInputTypeInfo<detect::BBoxResult>();
    this->setOutputTypeInfo<MOTResult>();
  }
  virtual ~BotSortNode() {}

  virtual base::Status init();
  virtual base::Status deinit();
  virtual base::Status run();

 private:
  std::unique_ptr<BotSORT> tracker_;
};

}  // namespace track
}  // namespace nndeploy

#endif  // _NNDEPLOY_TRACK_BOTSORT_BOT_SORT_NODE_H_
