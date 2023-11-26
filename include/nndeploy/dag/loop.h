#ifndef _NNDEPLOY_DAG_LOOP_H_
#define _NNDEPLOY_DAG_LOOP_H_

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
#include "nndeploy/dag/util.h"
#include "nndeploy/device/buffer.h"
#include "nndeploy/device/buffer_pool.h"
#include "nndeploy/device/device.h"
#include "nndeploy/device/tensor.h"

namespace nndeploy {
namespace dag {

class NNDEPLOY_CC_API Loop : public Node {
 public:
  Loop(const std::string& name, Edge* input, Edge* output);
  Loop(const std::string& name, std::initializer_list<Edge*> inputs,
       std::initializer_list<Edge*> outputs);
  virtual ~Loop();

  template <typename T,
            typename std::enable_if<std::is_base_of<Node, T>{}, int>::type = 0>
  Node* createNode(const std::string& name, Edge* input, Edge* output) {
    if (loop_node_ != nullptr) {
      NNDEPLOY_LOGE("loop_node_ must be nullptr!\n");
      return nullptr;
    }
    NNDEPLOY_CHECK_PARAM_NULL_RET_NULL(input, "input is null!");
    NNDEPLOY_CHECK_PARAM_NULL_RET_NULL(output, "output is null!");
    bool flag = checkEdge({input}, inputs_);
    if (!flag) {
      NNDEPLOY_LOGE("input is not in loop inputs!\n");
      return nullptr;
    }
    flag = checkEdge({output}, outputs_);
    if (!flag) {
      NNDEPLOY_LOGE("output is not in loop outputs!\n");
      return nullptr;
    }
    loop_node_ = dynamic_cast<Node*>(new T(name, input, output));
    return loop_node_;
  }
  template <typename T,
            typename std::enable_if<std::is_base_of<Node, T>{}, int>::type = 0>
  Node* createNode(const std::string& name, std::initializer_list<Edge*> inputs,
                   std::initializer_list<Edge*> outputs) {
    if (loop_node_ != nullptr) {
      NNDEPLOY_LOGE("loop_node_ must be nullptr!\n");
      return nullptr;
    }
    bool flag = checkEdge(inputs, inputs_);
    if (!flag) {
      NNDEPLOY_LOGE("inputs is not in loop inputs!\n");
      return nullptr;
    }
    flag = checkEdge(outputs, outputs_);
    if (!flag) {
      NNDEPLOY_LOGE("outputs is not in loop outputs!\n");
      return nullptr;
    }
    loop_node_ = dynamic_cast<Node*>(new T(name, inputs, outputs));
    return loop_node_;
  }
  template <typename T,
            typename std::enable_if<std::is_base_of<Node, T>{}, int>::type = 0>
  Node* createInfer(const std::string& name, base::InferenceType type,
                    Edge* input, Edge* output) {
    if (loop_node_ != nullptr) {
      NNDEPLOY_LOGE("loop_node_ must be nullptr!\n");
      return nullptr;
    }
    NNDEPLOY_CHECK_PARAM_NULL_RET_NULL(input, "input is null!");
    NNDEPLOY_CHECK_PARAM_NULL_RET_NULL(output, "output is null!");
    bool flag = checkEdge({input}, inputs_);
    if (!flag) {
      NNDEPLOY_LOGE("input is not in loop inputs!\n");
      return nullptr;
    }
    flag = checkEdge({output}, outputs_);
    if (!flag) {
      NNDEPLOY_LOGE("output is not in loop outputs!\n");
      return nullptr;
    }
    loop_node_ = dynamic_cast<Node*>(new T(name, type, input, output));
    return loop_node_;
  }
  template <typename T,
            typename std::enable_if<std::is_base_of<Node, T>{}, int>::type = 0>
  Node* createInfer(const std::string& name, base::InferenceType type,
                    std::initializer_list<Edge*> inputs,
                    std::initializer_list<Edge*> outputs) {
    if (loop_node_ != nullptr) {
      NNDEPLOY_LOGE("loop_node_ must be nullptr!\n");
      return nullptr;
    }
    bool flag = checkEdge(inputs, inputs_);
    if (!flag) {
      NNDEPLOY_LOGE("inputs is not in loop inputs!\n");
      return nullptr;
    }
    flag = checkEdge(outputs, outputs_);
    if (!flag) {
      NNDEPLOY_LOGE("outputs is not in loop outputs!\n");
      return nullptr;
    }
    loop_node_ = dynamic_cast<Node*>(new T(name, type, inputs, outputs));
    return loop_node_;
  }

  virtual base::Status init();
  virtual base::Status deinit();

  virtual int loops() = 0;

  virtual base::Status run();

 protected:
  Node* loop_node_ = nullptr;
};

}  // namespace dag
}  // namespace nndeploy

#endif /* _NNDEPLOY_DAG_LOOP_H_ */