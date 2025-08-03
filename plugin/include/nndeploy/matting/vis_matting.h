#ifndef _NNDEPLOY_MATTING_VISUALIZE_MATTING_H_
#define _NNDEPLOY_MATTING_VISUALIZE_MATTING_H_

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
#include "nndeploy/matting/result.h"

namespace nndeploy {
namespace matting {

class NNDEPLOY_CC_API VisMatting : public dag::Node {
 public:
  VisMatting(const std::string &name) : Node(name) {
    key_ = "nndeploy::matting::VisMatting";
    desc_ =
        "Draw matting result on input cv::Mat image based on matting "
        "results[cv::Mat->cv::Mat]";
    this->setInputTypeInfo<cv::Mat>();
    this->setInputTypeInfo<MattingResult>();
    this->setOutputTypeInfo<cv::Mat>();
  }
  VisMatting(const std::string &name, std::vector<dag::Edge *> inputs,
             std::vector<dag::Edge *> outputs)
      : Node(name, inputs, outputs) {
    key_ = "nndeploy::matting::VisMatting";
    desc_ =
        "Draw matting result on input cv::Mat image based on matting "
        "results[cv::Mat->cv::Mat]";
    this->setInputTypeInfo<cv::Mat>();
    this->setInputTypeInfo<MattingResult>();
    this->setOutputTypeInfo<cv::Mat>();
  }

  virtual ~VisMatting() {}

  virtual base::Status run();
};

}  // namespace matting
}  // namespace nndeploy

#endif