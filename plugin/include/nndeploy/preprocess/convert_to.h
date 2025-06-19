#ifndef _NNDEPLOY_PREPROCESS_CONVERT_TO_H_
#define _NNDEPLOY_PREPROCESS_CONVERT_TO_H_

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
#include "nndeploy/device/tensor.h"

namespace nndeploy {
namespace preprocess {

class NNDEPLOY_CC_API ConvertToParam : public base::Param {
 public:
  base::DataType dst_data_type_ = base::dataTypeOf<float>();

  using base::Param::serialize;
  virtual base::Status serialize(
      rapidjson::Value& json,
      rapidjson::Document::AllocatorType& allocator) override {
    json.AddMember("dst_data_type_",
                   rapidjson::Value(base::dataTypeToString(dst_data_type_).c_str(),
                                    base::dataTypeToString(dst_data_type_).length(),
                                    allocator),
                   allocator);
    return base::kStatusCodeOk;
  }

  using base::Param::deserialize;
  virtual base::Status deserialize(rapidjson::Value& json) override {
    if (json.HasMember("dst_data_type_") && json["dst_data_type_"].IsString()) {
      std::string dst_data_type_str = json["dst_data_type_"].GetString();
      dst_data_type_ = base::stringToDataType(dst_data_type_str);
    }
    return base::kStatusCodeOk;
  }
};

class NNDEPLOY_CC_API ConvertTo : public dag::Node {
 public:
  ConvertTo(const std::string &name) : dag::Node(name) {
    key_ = "nndeploy::preprocess::ConvertTo";
    desc_ = "Convert the data type of the input tensor to the specified data type";
    param_ = std::make_shared<ConvertToParam>();
    this->setInputTypeInfo<device::Tensor>();
    this->setOutputTypeInfo<device::Tensor>();
  }
  ConvertTo(const std::string &name, std::vector<dag::Edge *> inputs,
            std::vector<dag::Edge *> outputs)
      : dag::Node(name, inputs, outputs) {
    key_ = "nndeploy::preprocess::ConvertTo";
    desc_ = "Convert the data type of the input tensor to the specified data type";
    param_ = std::make_shared<ConvertToParam>();
    this->setInputTypeInfo<device::Tensor>();
    this->setOutputTypeInfo<device::Tensor>();
  }
  virtual ~ConvertTo() {}

  virtual base::Status run();
};

}  // namespace preprocess
}  // namespace nndeploy

#endif /* _NNDEPLOY_PREPROCESS_CONVERT_TO_H_ */
