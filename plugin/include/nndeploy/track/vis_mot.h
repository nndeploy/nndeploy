#ifndef _NNDEPLOY_TRACK_VISUALIZE_MOT_H_
#define _NNDEPLOY_TRACK_VISUALIZE_MOT_H_

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
#include "nndeploy/track/result.h"

namespace nndeploy {
namespace track {

class NNDEPLOY_CC_API VisMOTNode : public dag::Node {
 public:
  VisMOTNode(const std::string &name) : Node(name) {
    key_ = "nndeploy::track::VisMOTNode";
    this->setInputTypeInfo<cv::Mat>();
    this->setInputTypeInfo<MOTResult>();
    this->setOutputTypeInfo<cv::Mat>();
  }
  VisMOTNode(const std::string &name, std::vector<dag::Edge *> inputs,
             std::vector<dag::Edge *> outputs)
      : Node(name, inputs, outputs) {
    key_ = "nndeploy::track::VisMOTNode";
    this->setInputTypeInfo<cv::Mat>();
    this->setInputTypeInfo<MOTResult>();
    this->setOutputTypeInfo<cv::Mat>();
  }
  virtual ~VisMOTNode() {}

  virtual base::Status run();

  cv::Scalar GetMOTBoxColor(int idx);
};

}  // namespace track
}  // namespace nndeploy

#endif