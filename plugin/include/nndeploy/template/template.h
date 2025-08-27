#ifndef _NNDEPLOY_TEMPLATE_TEMPLATE_H_
#define _NNDEPLOY_TEMPLATE_TEMPLATE_H_

#include "nndeploy/base/any.h"
#include "nndeploy/base/common.h"
#include "nndeploy/base/glic_stl_include.h"
#include "nndeploy/base/log.h"
#include "nndeploy/base/macro.h"
#include "nndeploy/base/object.h"
#include "nndeploy/base/opencv_include.h"
#include "nndeploy/base/param.h"
#include "nndeploy/base/status.h"
#include "nndeploy/base/string.h"
#include "nndeploy/classification/result.h"
#include "nndeploy/dag/edge.h"
#include "nndeploy/dag/graph.h"
#include "nndeploy/dag/node.h"
#include "nndeploy/device/buffer.h"
#include "nndeploy/device/device.h"
#include "nndeploy/device/memory_pool.h"
#include "nndeploy/device/tensor.h"

namespace nndeploy {
namespace template_impl {

class TemplateParam : public base::Param {
 public:
  std::string template_param_ = "";

  using base::Param::serialize;
  virtual base::Status serialize(
      rapidjson::Value &json, rapidjson::Document::AllocatorType &allocator) {
    json.AddMember("template_param_",
                   rapidjson::Value(template_param_.c_str(),
                                    template_param_.length(), allocator),
                   allocator);
    return base::kStatusCodeOk;
  }
  using base::Param::deserialize;
  virtual base::Status deserialize(rapidjson::Value &json) {
    if (json.HasMember("template_param_") &&
        json["template_param_"].IsString()) {
      template_param_ = json["template_param_"].GetString();
    }
    return base::kStatusCodeOk;
  }
};

class NNDEPLOY_CC_API TemplateNode : public dag::Node {
 public:
  TemplateNode(const std::string &name, std::vector<dag::Edge *> inputs,
               std::vector<dag::Edge *> outputs)
      : dag::Node(name, inputs, outputs) {
    this->setKey("nndeploy::template_impl::TemplateNode");
    this->setDesc("Template node");
    this->setInputTypeInfo<device::Tensor>();
    this->setOutputTypeInfo<device::Tensor>();
    param_ = std::make_shared<TemplateParam>();
  }
  virtual ~TemplateNode() {}

  // tensor copy to tensor
  virtual base::Status run() {
    TemplateParam *tmp_param = dynamic_cast<TemplateParam *>(param_.get());
    device::Tensor *src = inputs_[0]->getTensor(this);
    std::string template_param = tmp_param->template_param_;
    device::Tensor *dst = outputs_[0]->create(src->getDevice(), src->getDesc());
    src->copyTo(dst);
    outputs_[0]->notifyWritten(dst);
    return base::kStatusCodeOk;
  }
};

class NNDEPLOY_CC_API TemplateGraph : public dag::Graph {
 public:
  TemplateGraph(const std::string &name, std::vector<dag::Edge *> inputs,
                std::vector<dag::Edge *> outputs  )
      : dag::Graph(name, inputs, outputs) {
    this->setKey("nndeploy::template_impl::TemplateGraph");
    this->setDesc("Template graph");
    this->setInputTypeInfo<device::Tensor>();
    this->setOutputTypeInfo<device::Tensor>();

    pre_ = dynamic_cast<TemplateNode *>(this->createNode<TemplateNode>("pre"));
    infer_ = dynamic_cast<TemplateNode *>(this->createNode<TemplateNode>("infer"));
    post_ = dynamic_cast<TemplateNode *>(this->createNode<TemplateNode>("post"));
  }
  virtual ~TemplateGraph() {}

  // default param
  virtual base::Status defaultParam() { return base::kStatusCodeOk; }

  // static graph
  base::Status make(dag::NodeDesc &pre_desc, dag::NodeDesc &infer_desc,
                    dag::NodeDesc &post_desc) {
    this->setNodeDesc(pre_, pre_desc);
    this->setNodeDesc(infer_, infer_desc);
    this->setNodeDesc(post_, post_desc);
    return base::kStatusCodeOk;
  }

  // dynamic graph
  std::vector<dag::Edge *> forward(std::vector<dag::Edge *> &inputs) {
    std::vector<dag::Edge *> outputs;
    std::vector<dag::Edge *> pre_outputs = (*pre_)(inputs);
    std::vector<dag::Edge *> infer_outputs = (*infer_)(pre_outputs[0]);
    std::vector<dag::Edge *> post_outputs = (*post_)(infer_outputs[0]);
    return post_outputs;
  }

 private:
  TemplateNode *pre_ = nullptr;
  TemplateNode *infer_ = nullptr;
  TemplateNode *post_ = nullptr;
};

}  // namespace template_impl
}  // namespace nndeploy

#endif /* _NNDEPLOY_TEMPLATE_TEMPLATE_H_ */