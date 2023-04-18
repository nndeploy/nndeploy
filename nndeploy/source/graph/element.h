#ifndef _NNDEPLOY_SOURCE_GRAPH_BASE_H_
#define _NNDEPLOY_SOURCE_GRAPH_BASE_H_

#include "nndeploy/source/base/basic.h"
#include "nndeploy/source/base/glic_stl_include.h"
#include "nndeploy/source/base/log.h"
#include "nndeploy/source/base/macro.h"
#include "nndeploy/source/base/object.h"
#include "nndeploy/source/base/status.h"

namespace nndeploy {
namespace graph {

class Element {
 public:
  Element() = default;
  virtual ~Element() = default;

  const std::string& getGuid() const { return guid_; }
  void setGuid(const std::string& guid) { guid_ = guid; }

  const std::string& getName() const { return name_; }
  void setName(const std::string& name) { name_ = name; }

  const uint32_t getLoopCount() const { return loop_count_; }
  void setLoopCount(uint32_t loop_count) { loop_count_ = loop_count; }

  virtaul base::Status setParam(Param* param) {
    params_.push_back(param);
    return base::kStatusCodeOk;
  }
  virtual base::Status setParam(std::vector<Param*> param) {
    params_.insert(params_.end(), param.begin(), param.end());
    return base::kStatusCodeOk;
  }

  virtual base::Status setInput(Edge* edge) {
    inputs_.push_back(edge);
    return base::kStatusCodeOk;
  }
  virtual base::Status setInput(std::vector<Edge*> edge) {
    inputs_.insert(inputs_.end(), edge.begin(), edge.end());
    return base::kStatusCodeOk;
  }
  virtual base::Status setOutput(Edge* edge) {
    inputs_.push_back(edge);
    return base::kStatusCodeOk;
  }
  virtual base::Status setOutput(std::vector<Edge*> edge) {
    outputs_.insert(outputs_.end(), edge.begin(), edge.end());
    return base::kStatusCodeOk;
  }

  virtual base::Status setDependElement(Element* element) {
    depend_elements_.push_back(element);
    return base::kStatusCodeOk;
  }
  virtual base::Status setDependElement(std::vector<Element*> element) {
    depend_elements_.insert(depend_elements_.end(), element.begin(),
                            element.end());
    return base::kStatusCodeOk;
  }

  virtual base::Status init() { return base::kStatusCodeOk; }
  virtual base::Status deinit() { return base::kStatusCodeOk; }

  virtual base::Status preRun() { return base::kStatusCodeOk; }
  virtual base::Status postRun() { return base::kStatusCodeOk; }

  virtual base::Status Run() { return base::kStatusCodeOk; }
  virtual base::Status aysncRun() { return base::kStatusCodeOk; }

 protected:
  std::string guid_;  // 唯一id信息
  std::string name_;  // 名字

  uint32_t loop_count_ = 1;

  std::vector<Param*> params_;

  std::vector<Edge*> inputs_;
  std::vector<Edge*> outputs_;

  std::vector<Element*> depend_elements_;
};

}  // namespace graph
}  // namespace nndeploy

#endif /* _NNDEPLOY_SOURCE_GRAPH_BASE_H_ */
