#ifndef _NNDEPLOY_SEGMENT_DRAWMASK_H_
#define _NNDEPLOY_SEGMENT_DRAWMASK_H_

#include "nndeploy/base/glic_stl_include.h"
#include "nndeploy/base/time_profiler.h"
#include "nndeploy/dag/node.h"
#include "nndeploy/device/device.h"
#include "nndeploy/segment/result.h"
#include "nndeploy/thread_pool/thread_pool.h"

namespace nndeploy {
namespace segment {

class NNDEPLOY_CC_API DrawMask : public dag::Node {
 public:
  DrawMask(const std::string &name) : Node(name) {
    key_ = "nndeploy::segment::DrawMask";
    desc_ =
        "Draw segmentation mask on input cv::Mat image based on segmentation "
        "results[cv::Mat->cv::Mat]";
    this->setInputTypeInfo<cv::Mat>();
    this->setInputTypeInfo<SegmentResult>();
    this->setOutputTypeInfo<cv::Mat>();
  }
  DrawMask(const std::string &name, std::vector<dag::Edge *> inputs,
           std::vector<dag::Edge *> outputs)
      : Node(name, inputs, outputs) {
    key_ = "nndeploy::segment::DrawMask";
    desc_ =
        "Draw segmentation mask on input cv::Mat image based on segmentation "
        "results[cv::Mat->cv::Mat]";
    this->setInputTypeInfo<cv::Mat>();
    this->setInputTypeInfo<SegmentResult>();
    this->setOutputTypeInfo<cv::Mat>();
  }
  virtual ~DrawMask() {}

  virtual base::Status run();
};

}  // namespace segment
}  // namespace nndeploy

#endif