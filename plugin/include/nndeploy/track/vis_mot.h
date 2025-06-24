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

class NNDEPLOY_CC_API VisMOT : public dag::Node {
 public:
  VisMOT(const std::string &name) : Node(name) {
    key_ = "nndeploy::track::VisMOT";
    desc_ =
        "Draw MOT result on input cv::Mat image based on MOT results[cv::Mat->cv::Mat]";
    this->setInputTypeInfo<cv::Mat>();
    this->setInputTypeInfo<MOTResult>();
    this->setOutputTypeInfo<cv::Mat>();
  }
  VisMOT(const std::string &name, std::vector<dag::Edge *> inputs,
             std::vector<dag::Edge *> outputs)
      : Node(name, inputs, outputs) {
    key_ = "nndeploy::track::VisMOT";
    desc_ =
        "Draw MOT result on input cv::Mat image based on MOT results[cv::Mat->cv::Mat]";
    this->setInputTypeInfo<cv::Mat>();
    this->setInputTypeInfo<MOTResult>();
    this->setOutputTypeInfo<cv::Mat>();
  }
  virtual ~VisMOT() {}

  virtual base::Status run();

  cv::Scalar GetMOTBoxColor(int idx);
};

}  // namespace track
}  // namespace nndeploy

#endif