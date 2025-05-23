#ifndef _NNDEPLOY_PREPROCESS_BATCH_PREPROCESS_H_
#define _NNDEPLOY_PREPROCESS_BATCH_PREPROCESS_H_

#include "nndeploy/base/any.h"
#include "nndeploy/base/common.h"
#include "nndeploy/base/glic_stl_include.h"
#include "nndeploy/base/log.h"
#include "nndeploy/base/macro.h"
#include "nndeploy/base/object.h"
#include "nndeploy/base/opencv_include.h"
#include "nndeploy/base/status.h"
#include "nndeploy/base/string.h"
#include "nndeploy/dag/edge.h"
#include "nndeploy/dag/node.h"
#include "nndeploy/device/buffer.h"
#include "nndeploy/device/device.h"
#include "nndeploy/device/memory_pool.h"
#include "nndeploy/device/tensor.h"
#include "nndeploy/preprocess/opencv_convert.h"
#include "nndeploy/preprocess/params.h"
#include "nndeploy/dag/composite_node.h"

namespace nndeploy {
namespace preprocess {

class NNDEPLOY_CC_API BatchPreprocess : public dag::CompositeNode {
 public:
  BatchPreprocess(const std::string &name) : dag::CompositeNode(name) {
    key_ = "nndeploy::preprocess::BatchPreprocess";
    this->setInputTypeInfo<std::vector<cv::Mat>>();
    this->setOutputTypeInfo<device::Tensor>();
  }
  BatchPreprocess(const std::string &name, std::vector<dag::Edge *> inputs,
                 std::vector<dag::Edge *> outputs)
      : dag::CompositeNode(name, inputs, outputs) {
    key_ = "nndeploy::preprocess::BatchPreprocess";
    this->setInputTypeInfo<std::vector<cv::Mat>>();
    this->setOutputTypeInfo<device::Tensor>();
  }
  virtual ~BatchPreprocess() {}

  base::ParamList getParams() override {
    base::ParamList params;
    params.push_back(base::Param("data_format", data_format_));
    return params;
  }

  base::Status setNodeKey(const std::string &key);

  virtual base::Status make();

  virtual base::Status run();

 private:
  base::DataFormat data_format_ = base::kDataFormatNCHW;
  std::string node_key_ = "";
  dag::Node *node_ = nullptr;
};

}  // namespace preprocess
}  // namespace nndeploy

#endif /* _NNDEPLOY_PREPROCESS_BATCH_PREPROCESS_H_ */
