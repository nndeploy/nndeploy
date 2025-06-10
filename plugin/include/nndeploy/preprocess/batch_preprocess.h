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

  base::Status setNodeKey(const std::string &key);
  base::Status setDataFormat(base::DataFormat data_format);
  base::DataFormat getDataFormat();

  virtual base::Status setParam(base::Param *param);
  virtual base::Status setParamSharedPtr(std::shared_ptr<base::Param> param);
  virtual base::Param *getParam();
  virtual std::shared_ptr<base::Param> getParamSharedPtr();

  virtual base::Status run();

  virtual base::Status serialize(rapidjson::Value &json,
                                 rapidjson::Document::AllocatorType &allocator);
  virtual std::string serialize();
  virtual base::Status deserialize(rapidjson::Value &json);
  virtual base::Status deserialize(const std::string &json_str);

 private:
  base::DataFormat data_format_ = base::kDataFormatNCHW;
  std::string node_key_ = "";
  dag::Node *node_ = nullptr;
};

}  // namespace preprocess
}  // namespace nndeploy

#endif /* _NNDEPLOY_PREPROCESS_BATCH_PREPROCESS_H_ */
