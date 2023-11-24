#ifndef _NNDEPLOY_DAG_CONDITION_H_
#define _NNDEPLOY_DAG_CONDITION_H_

#include "nndeploy/base/common.h"
#include "nndeploy/base/glic_stl_include.h"
#include "nndeploy/base/log.h"
#include "nndeploy/base/macro.h"
#include "nndeploy/base/object.h"
#include "nndeploy/base/status.h"
#include "nndeploy/base/string.h"
#include "nndeploy/base/value.h"
#include "nndeploy/dag/edge.h"
#include "nndeploy/dag/node.h"
#include "nndeploy/device/buffer.h"
#include "nndeploy/device/buffer_pool.h"
#include "nndeploy/device/device.h"
#include "nndeploy/device/tensor.h"

namespace nndeploy {
namespace dag {

class NNDEPLOY_CC_API Condition : public Node {
 public:
  Condition(const std::string& name, Edge* input, Edge* output);
  Condition(const std::string& name, std::initializer_list<Edge*> inputs,
            std::initializer_list<Edge*> outputs);
  virtual ~Condition();

  template <typename T,
            typename std::enable_if<std::is_base_of<Node, T>{}, int>::type = 0>
  Node* createNode(const std::string& name, Edge* input, Edge* output) {
    NNDEPLOY_CHECK_PARAM_NULL_RET_NULL(input, "input is null!");
    NNDEPLOY_CHECK_PARAM_NULL_RET_NULL(output, "output is null!");
    bool flag = check({input}, inputs_);
    if (!flag) {
      NNDEPLOY_LOGE("input is not in condition inputs!\n");
      return nullptr;
    }
    flag = check({output}, outputs_);
    if (!flag) {
      NNDEPLOY_LOGE("output is not in condition outputs!\n");
      return nullptr;
    }
    Node* node = dynamic_cast<Node*>(new T(name, input, output));
    condition_node_.emplace_back(node);
    return node;
  }
  template <typename T,
            typename std::enable_if<std::is_base_of<Node, T>{}, int>::type = 0>
  Node* createNode(const std::string& name, std::initializer_list<Edge*> inputs,
                   std::initializer_list<Edge*> outputs) {
    bool flag = check(inputs, inputs_);
    if (!flag) {
      NNDEPLOY_LOGE("inputs is not in condition inputs!\n");
      return nullptr;
    }
    flag = check(outputs, outputs_);
    if (!flag) {
      NNDEPLOY_LOGE("outputs is not in condition outputs!\n");
      return nullptr;
    }
    Node* node = dynamic_cast<Node*>(new T(name, inputs, outputs));
    condition_node_.emplace_back(node);
    return node;
  }
  template <typename T,
            typename std::enable_if<std::is_base_of<Node, T>{}, int>::type = 0>
  Node* createInfer(const std::string& name, base::InferenceType type,
                    Edge* input, Edge* output) {
    NNDEPLOY_CHECK_PARAM_NULL_RET_NULL(input, "input is null!");
    NNDEPLOY_CHECK_PARAM_NULL_RET_NULL(output, "output is null!");
    bool flag = check({input}, inputs_);
    if (!flag) {
      NNDEPLOY_LOGE("input is not in condition inputs!\n");
      return nullptr;
    }
    flag = check({output}, outputs_);
    if (!flag) {
      NNDEPLOY_LOGE("output is not in condition outputs!\n");
      return nullptr;
    }
    Node* node = dynamic_cast<Node*>(new T(name, type, input, output));
    condition_node_.emplace_back(node);
    return node;
  }
  template <typename T,
            typename std::enable_if<std::is_base_of<Node, T>{}, int>::type = 0>
  Node* createInfer(const std::string& name, base::InferenceType type,
                    std::initializer_list<Edge*> inputs,
                    std::initializer_list<Edge*> outputs) {
    bool flag = check(inputs, inputs_);
    if (!flag) {
      NNDEPLOY_LOGE("inputs is not in condition inputs!\n");
      return nullptr;
    }
    flag = check(outputs, outputs_);
    if (!flag) {
      NNDEPLOY_LOGE("outputs is not in condition outputs!\n");
      return nullptr;
    }
    Node* node = dynamic_cast<Node*>(new T(name, type, inputs, outputs));
    condition_node_.emplace_back(node);
    return node;
  }

  base::Status setNodeParam(const std::string& node_name, base::Param* param);
  base::Param* getNodeParam(const std::string& node_name);

  virtual base::Status init();
  virtual base::Status deinit();

  virtual int choose() = 0;

  virtual base::Status run();

 private:
  bool check(const std::vector<Edge*>& edges,
             const std::vector<Edge*>& condition_edges);
  Node* findNode(const std::string& name);

 protected:
  std::vector<Node*> condition_node_;
};

}  // namespace dag
}  // namespace nndeploy

#endif /* _NNDEPLOY_DAG_CONDITION_H_ */