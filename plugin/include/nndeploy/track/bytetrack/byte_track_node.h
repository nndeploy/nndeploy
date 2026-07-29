#ifndef _NNDEPLOY_TRACK_BYTETRACK_BYTE_TRACK_NODE_H_
#define _NNDEPLOY_TRACK_BYTETRACK_BYTE_TRACK_NODE_H_

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
#include "nndeploy/track/bytetrack/bytetrack.h"
#include "nndeploy/track/result.h"

namespace nndeploy {
namespace track {

class NNDEPLOY_CC_API ByteTrackParam : public base::Param {
 public:
  float track_thresh_ = 0.5f;
  float high_thresh_ = 0.6f;
  float match_thresh_ = 0.8f;
  int max_lost_time_ = 30;
  int frame_rate_ = 30;

  virtual base::Status serialize(rapidjson::Value &json,
                                 rapidjson::Document::AllocatorType &allocator);
  virtual base::Status deserialize(rapidjson::Value &json);
};

/**
 * @brief ByteTrack DAG node
 *
 * Wraps the ByteTrack tracker as a DAG node.
 * Input:  DetectResult (from a detection graph/node)
 * Output: MOTResult   (tracked objects with IDs)
 *
 * The node maintains persistent tracking state across consecutive
 * run() calls. Use reset() or set param track_thresh_ = 0 to reinitialize.
 */
class NNDEPLOY_CC_API ByteTrackNode : public dag::Node {
 public:
  ByteTrackNode(const std::string &name) : Node(name) {
    key_ = "nndeploy::track::ByteTrackNode";
    desc_ = "ByteTrack multi-object tracker[BBoxResult->MOTResult]";
    param_ = std::make_shared<ByteTrackParam>();
    this->setInputTypeInfo<detect::BBoxResult>();
    this->setOutputTypeInfo<MOTResult>();
  }
  ByteTrackNode(const std::string &name, std::vector<dag::Edge *> inputs,
                std::vector<dag::Edge *> outputs)
      : Node(name, inputs, outputs) {
    key_ = "nndeploy::track::ByteTrackNode";
    desc_ = "ByteTrack multi-object tracker[BBoxResult->MOTResult]";
    param_ = std::make_shared<ByteTrackParam>();
    this->setInputTypeInfo<detect::BBoxResult>();
    this->setOutputTypeInfo<MOTResult>();
  }
  virtual ~ByteTrackNode() {}

  virtual base::Status init();
  virtual base::Status deinit();
  virtual base::Status run();

 private:
  std::unique_ptr<ByteTrack> tracker_;
};

}  // namespace track
}  // namespace nndeploy

#endif  // _NNDEPLOY_TRACK_BYTETRACK_BYTE_TRACK_NODE_H_
