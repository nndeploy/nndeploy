#ifndef _NNDEPLOY_DAG_GRAPH_H_
#define _NNDEPLOY_DAG_GRAPH_H_

#include "nndeploy/base/any.h"
#include "nndeploy/base/common.h"
#include "nndeploy/base/glic_stl_include.h"
#include "nndeploy/base/log.h"
#include "nndeploy/base/macro.h"
#include "nndeploy/base/object.h"
#include "nndeploy/base/status.h"
#include "nndeploy/base/string.h"
#include "nndeploy/dag/edge.h"
#include "nndeploy/dag/executor.h"
#include "nndeploy/dag/node.h"
#include "nndeploy/dag/util.h"
#include "nndeploy/device/buffer.h"
#include "nndeploy/device/device.h"
#include "nndeploy/device/memory_pool.h"
#include "nndeploy/device/tensor.h"
#include "nndeploy/inference/inference_param.h"
/**
 * @brief 有向无环图
 */

namespace nndeploy {
namespace dag {

/**
 * @brief 有向无环图节点
 */
class NNDEPLOY_CC_API Graph : public Node {
 public:
  Graph(const std::string &name);
  Graph(const std::string &name, std::vector<Edge *> inputs,
        std::vector<Edge *> outputs);

  virtual ~Graph();

  /**
   * @brief 在Graph中创建一条Edge
   * @param  name             edge名称
   * @return Edge*
   */
  Edge *createEdge(const std::string &name);

  std::shared_ptr<Edge> createEdgeSharedPtr(const std::string &name);

  /**
   * @brief 将一条已有的Edge加入Graph中
   * @param  edge             已有的Edge
   * @return EdgeWrapper*
   */
  // EdgeWrapper *addEdge(Edge *edge);

  /**
   * @brief 将一条已有的Edge加入Graph中
   * @param  edge             已有的Edge
   * @return EdgeWrapper*
   */
  EdgeWrapper *addEdge(Edge *edge, bool is_external = true);
  EdgeWrapper *addEdgeSharedPtr(std::shared_ptr<Edge> edge);

  base::Status updteEdge(EdgeWrapper *edge_wrapper, Edge *edge,
                         bool is_external = true);

  /**
   * @brief 获取Graph中的Edge
   * @param  name             edge名称
   * @return Edge*
   */
  Edge *getEdge(const std::string &name);
  std::shared_ptr<Edge> getEdgeSharedPtr(const std::string &name);

  /**
   * @brief 在Graph中创建一个Node
   * @param  name             Node名称
   * @param  args             Node的参数
   * @return Node*
   */
  template <typename T, typename... Args,
            typename std::enable_if<std::is_base_of<Node, T>{}, int>::type = 0>
  Node *createNode(const std::string &name, Args &...args);

  /**
   * @brief 在Graph中创建一个Node，并关联input、output的Edge
   * @param  name             Node名称
   * @param  input            输入Edge
   * @param  output           输出Edge
   * @return Node*
   */
  template <typename T, typename... Args,
            typename std::enable_if<std::is_base_of<Node, T>{}, int>::type = 0>
  Node *createNode(const std::string &name, Edge *input, Edge *output,
                   Args &...args);

  /**
   * @brief 在Graph中创建一个Node，并关联input、output的Edge
   * @param  name             Node名称
   * @param  input            输入Edge name
   * @param  output           输出Edge name
   * @return Node*
   */
  template <typename T, typename... Args,
            typename std::enable_if<std::is_base_of<Node, T>{}, int>::type = 0>
  Node *createNode(const std::string &name, const std::string &input_name,
                   const std::string &output_name, Args &...args);

  /**
   * @brief 在Graph中创建一个Node，并关联input、output的Edge
   * @param  name             Node名称
   * @param  input            输入Edge
   * @param  output           输出Edge name
   * @return Node*
   */
  template <typename T, typename... Args,
            typename std::enable_if<std::is_base_of<Node, T>{}, int>::type = 0>
  Node *createNode(const std::string &name, Edge *input,
                   const std::string &output_name, Args &...args);

  /**
   * @brief 在Graph中创建一个Node，并关联input、output的Edge
   * @param  name             Node名称
   * @param  input            输入Edge name
   * @param  output           输出Edge
   * @return Node*
   */
  template <typename T, typename... Args,
            typename std::enable_if<std::is_base_of<Node, T>{}, int>::type = 0>
  Node *createNode(const std::string &name, const std::string &input_name,
                   Edge *output, Args &...args);

  /**
   * @brief 在Graph中创建一个Node，并关联多个input、output的Edge
   * @param  name             Node名称
   * @param  inputs            多个输入Edge
   * @param  outputs           多个输出Edge
   * @return Node*
   */
  template <typename T, typename... Args,
            typename std::enable_if<std::is_base_of<Node, T>{}, int>::type = 0>
  Node *createNode(const std::string &name, std::vector<Edge *> inputs,
                   std::vector<Edge *> outputs, Args &...args);

  /**
   * @brief 在Graph中创建一个Node，并关联多个input、output的Edge
   * @param  name             Node名称
   * @param  inputs            多个输入Edge
   * @param  outputs           多个输出Edge
   * @return Node*
   */
  template <typename T, typename... Args,
            typename std::enable_if<std::is_base_of<Node, T>{}, int>::type = 0>
  Node *createNode(const std::string &name,
                   std::vector<std::string> input_names,
                   std::vector<std::string> output_names, Args &...args);

  /**
   * @brief 在Graph中创建一个Node，并关联多个input、output的Edge
   * @param  name             Node名称
   * @param  inputs            多个输入Edge
   * @param  outputs           多个输出Edge
   * @return Node*
   */
  template <typename T, typename... Args,
            typename std::enable_if<std::is_base_of<Node, T>{}, int>::type = 0>
  Node *createNode(const std::string &name,
                   std::vector<std::string> input_names,
                   std::vector<Edge *> outputs, Args &...args);

  /**
   * @brief 在Graph中创建一个Node，并关联多个input、output的Edge
   * @param  name             Node名称
   * @param  inputs            多个输入Edge
   * @param  outputs           多个输出Edge
   * @return Node*
   */
  template <typename T, typename... Args,
            typename std::enable_if<std::is_base_of<Node, T>{}, int>::type = 0>
  Node *createNode(const std::string &name, std::vector<Edge *> inputs,
                   std::vector<std::string> output_names, Args &...args);

  /**
   * @brief 在Graph中创建一个Node，并关联多个input、output的Edge
   * @param  name             Node名称
   * @param  inputs            多个输入Edge
   * @param  outputs           多个输出Edge
   * @return Node*
   */
  template <typename T, typename... Args,
            typename std::enable_if<std::is_base_of<Node, T>{}, int>::type = 0>
  Node *createNode(const std::string &name,
                   std::initializer_list<Edge *> inputs,
                   std::initializer_list<Edge *> outputs, Args &...args);

  /**
   * @brief 在Graph中创建一个Node，并关联多个input、output的Edge
   * @param  name             Node名称
   * @param  inputs            多个输入Edge name
   * @param  outputs           多个输出Edge name
   * @return Node*
   */
  template <typename T, typename... Args,
            typename std::enable_if<std::is_base_of<Node, T>{}, int>::type = 0>
  Node *createNode(const std::string &name,
                   std::initializer_list<std::string> input_names,
                   std::initializer_list<std::string> output_names,
                   Args &...args);

  /**
   * @brief 在Graph中创建一个Node，并关联多个input、output的Edge
   * @param  name             Node名称
   * @param  inputs            多个输入Edge
   * @param  outputs           多个输出Edge name
   * @return Node*
   */
  template <typename T, typename... Args,
            typename std::enable_if<std::is_base_of<Node, T>{}, int>::type = 0>
  Node *createNode(const std::string &name,
                   std::initializer_list<Edge *> inputs,
                   std::initializer_list<std::string> output_names,
                   Args &...args);

  /**
   * @brief 在Graph中创建一个Node，并关联多个input、output的Edge
   * @param  name             Node名称
   * @param  inputs            多个输入Edge name
   * @param  outputs           多个输出Edge
   * @return Node*
   */
  template <typename T, typename... Args,
            typename std::enable_if<std::is_base_of<Node, T>{}, int>::type = 0>
  Node *createNode(const std::string &name,
                   std::initializer_list<std::string> input_names,
                   std::initializer_list<Edge *> outputs, Args &...args);

  /**
   * @brief 在Graph中创建一个Infer Node，并关联input、output的Edge
   * @param  name             Node名称
   * @param  type            Infer的引擎类型
   * @param  input            输入Edge
   * @param  output           输出Edge
   * @return Node*
   */
  template <typename T, typename... Args,
            typename std::enable_if<std::is_base_of<Node, T>{}, int>::type = 0>
  Node *createInfer(const std::string &name, base::InferenceType type,
                    Edge *input, Edge *output);

  /**
   * @brief 在Graph中创建一个Infer Node，并关联input、output的Edge
   * @param  name             Node名称
   * @param  type            Infer的引擎类型
   * @param  input            输入Edge name
   * @param  output           输出Edge name
   * @return Node*
   */
  template <typename T, typename... Args,
            typename std::enable_if<std::is_base_of<Node, T>{}, int>::type = 0>
  Node *createInfer(const std::string &name, base::InferenceType type,
                    const std::string &input_name,
                    const std::string &output_name);

  /**
   * @brief 在Graph中创建一个Infer Node，并关联input、output的Edge
   * @param  name             Node名称
   * @param  type            Infer的引擎类型
   * @param  input            输入Edge name
   * @param  output           输出Edge name
   * @return Node*
   */
  template <typename T, typename... Args,
            typename std::enable_if<std::is_base_of<Node, T>{}, int>::type = 0>
  Node *createInfer(const std::string &name, base::InferenceType type,
                    Edge *input, const std::string &output_name);

  /**
   * @brief 在Graph中创建一个Infer Node，并关联input、output的Edge
   * @param  name             Node名称
   * @param  type            Infer的引擎类型
   * @param  input            输入Edge name
   * @param  output           输出Edge name
   * @return Node*
   */
  template <typename T, typename... Args,
            typename std::enable_if<std::is_base_of<Node, T>{}, int>::type = 0>
  Node *createInfer(const std::string &name, base::InferenceType type,
                    const std::string &input_name, Edge *output);

  /**
   * @brief 在Graph中创建一个Infer Node，并关联多个input、output的Edge
   * @param  name             Node名称
   * @param  type            Infer的引擎类型
   * @param  inputs            多个输入Edge
   * @param  outputs           多个输出Edge
   * @return Node*
   */
  template <typename T, typename... Args,
            typename std::enable_if<std::is_base_of<Node, T>{}, int>::type = 0>
  Node *createInfer(const std::string &name, base::InferenceType type,
                    std::vector<Edge *> inputs, std::vector<Edge *> outputs);

  /**
   * @brief 在Graph中创建一个Infer Node，并关联多个input、output的Edge
   * @param  name             Node名称
   * @param  type            Infer的引擎类型
   * @param  inputs            多个输入Edge name
   * @param  outputs           多个输出Edge name
   * @return Node*
   */
  template <typename T, typename... Args,
            typename std::enable_if<std::is_base_of<Node, T>{}, int>::type = 0>
  Node *createInfer(const std::string &name, base::InferenceType type,
                    std::vector<std::string> input_names,
                    std::vector<std::string> output_names);

  /**
   * @brief 在Graph中创建一个Infer Node，并关联多个input、output的Edge
   * @param  name             Node名称
   * @param  type            Infer的引擎类型
   * @param  inputs            多个输入Edge
   * @param  outputs           多个输出Edge name
   * @return Node*
   */
  template <typename T, typename... Args,
            typename std::enable_if<std::is_base_of<Node, T>{}, int>::type = 0>
  Node *createInfer(const std::string &name, base::InferenceType type,
                    std::vector<Edge *> inputs,
                    std::vector<std::string> output_names);

  /**
   * @brief 在Graph中创建一个Infer Node，并关联多个input、output的Edge
   * @param  name             Node名称
   * @param  type            Infer的引擎类型
   * @param  inputs            多个输入Edge name
   * @param  outputs           多个输出Edge name
   * @return Node*
   */
  template <typename T, typename... Args,
            typename std::enable_if<std::is_base_of<Node, T>{}, int>::type = 0>
  Node *createInfer(const std::string &name, base::InferenceType type,
                    std::vector<std::string> input_names,
                    std::vector<Edge *> outputs);

  /**
   * @brief 在Graph中创建一个Infer Node，并关联多个input、output的Edge
   * @param  name             Node名称
   * @param  type            Infer的引擎类型
   * @param  inputs            多个输入Edge
   * @param  outputs           多个输出Edge
   * @return Node*
   */
  template <typename T, typename... Args,
            typename std::enable_if<std::is_base_of<Node, T>{}, int>::type = 0>
  Node *createInfer(const std::string &name, base::InferenceType type,
                    std::initializer_list<Edge *> inputs,
                    std::initializer_list<Edge *> outputs);

  /**
   * @brief 在Graph中创建一个Infer Node，并关联多个input、output的Edge
   * @param  name             Node名称
   * @param  type            Infer的引擎类型
   * @param  inputs            多个输入Edge name
   * @param  outputs           多个输出Edge name
   * @return Node*
   */
  template <typename T, typename... Args,
            typename std::enable_if<std::is_base_of<Node, T>{}, int>::type = 0>
  Node *createInfer(const std::string &name, base::InferenceType type,
                    std::initializer_list<std::string> input_names,
                    std::initializer_list<std::string> output_names);

  /**
   * @brief 在Graph中创建一个Infer Node，并关联多个input、output的Edge
   * @param  name             Node名称
   * @param  type            Infer的引擎类型
   * @param  inputs            多个输入Edge
   * @param  outputs           多个输出Edge name
   * @return Node*
   */
  template <typename T, typename... Args,
            typename std::enable_if<std::is_base_of<Node, T>{}, int>::type = 0>
  Node *createInfer(const std::string &name, base::InferenceType type,
                    std::initializer_list<Edge *> inputs,
                    std::initializer_list<std::string> output_names);

  /**
   * @brief 在Graph中创建一个Infer Node，并关联多个input、output的Edge
   * @param  name             Node名称
   * @param  type            Infer的引擎类型
   * @param  inputs            多个输入Edge name
   * @param  outputs           多个输出Edge
   * @return Node*
   */
  template <typename T, typename... Args,
            typename std::enable_if<std::is_base_of<Node, T>{}, int>::type = 0>
  Node *createInfer(const std::string &name, base::InferenceType type,
                    std::initializer_list<std::string> input_names,
                    std::initializer_list<Edge *> outputs);

  template <typename T, typename... Args,
            typename std::enable_if<std::is_base_of<Node, T>{}, int>::type = 0>
  Node *createNode(const NodeDesc &desc, Args &...args);

  Node *createNodeByKey(const NodeDesc &desc);

  /**
   * @brief 将一个已有的Node加入Graph
   * @param  node             已有Node
   * @return base::Status
   */
  base::Status addNode(Node *node, bool is_external = true);
  base::Status addNodeSharedPtr(std::shared_ptr<Node> node);

  base::Status setNodeParam(const std::string &node_name, base::Param *param);
  base::Param *getNodeParam(const std::string &node_name);
  base::Status setNodeParamSharedPtr(const std::string &node_name,
                                     std::shared_ptr<base::Param> param);
  std::shared_ptr<base::Param> getNodeParamSharedPtr(
      const std::string &node_name);

  void setGraphNodeShareStream(bool flag);
  bool getGraphNodeShareStream();

  base::Status updateNodeIO(Node *node, std::vector<Edge *> inputs,
                            std::vector<Edge *> outputs);
  base::Status markInputEdge(std::vector<Edge *> inputs) {
    for (auto input : inputs) {
      insertUnique(inputs_, input);
    }
    return base::kStatusCodeOk;
  };
  base::Status markOutputEdge(std::vector<Edge *> outputs) {
    for (auto output : outputs) {
      // output->markGraphOutput();
      insertUnique(outputs_, output);
    }
    return base::kStatusCodeOk;
  };

  virtual base::Status init();
  virtual base::Status deinit();

  virtual base::Status run();

  // This method must be implemented by subclasses
  // Subclasses should override this method to define their own operator()
  // implementation
  virtual std::vector<Edge *> forward(std::vector<Edge *> inputs) {
    // check
    // if (!checkInputs(inputs)) {
    //   return std::vector<Edge *>();
    // }
    // if (!checkOutputs(outputs_name)) {
    //   return std::vector<Edge *>();
    // }
    // if (param != nullptr) {
    //   this->setParamSharedPtr(param);
    // }
    // bool is_inputs_changed = isInputsChanged(inputs);
    // if (!inputs.empty()) {
    //   this->setInputs(inputs);
    // }
    // std::vector<std::string> real_outputs_name = this->getRealOutputsName();
    // std::vector<Edge *> outputs;
    // for (auto name : real_outputs_name) {
    //   Edge *edge = nullptr;
    //   if (graph_ != nullptr) {
    //     edge = graph_->getEdge(name);
    //     if (edge != nullptr) {
    //       outputs.push_back(edge);
    //     }
    //   }
    //   if (edge == nullptr) {
    //     edge = this->createEdge(name);
    //     if (edge != nullptr) {
    //       outputs.push_back(edge);
    //     } else {
    //       NNDEPLOY_LOGE("createEdge failed.\n");
    //       return std::vector<Edge *>();
    //     }
    //   }
    // }
    // if (!outputs.empty()) {
    //   this->setOutputs(outputs);
    // }
    // if (graph_ != nullptr) {
    //   base::Status status = graph_->updateNodeIO(this, inputs, outputs);
    //   if (status != base::kStatusCodeOk) {
    //     NNDEPLOY_LOGE("graph_->updateNodeIO failed.\n");
    //     return std::vector<Edge *>();
    //   }
    // }
    // if (!is_inputs_changed && is_compiled_) {
    //   if (initialized_ == false) {
    //     this->init();
    //     this->setInitializedFlag(true);
    //   }
    //   base::Status status = this->run();
    //   if (status != base::kStatusCodeOk) {
    //     NNDEPLOY_LOGE("this->run() failed.\n");
    //     return std::vector<Edge *>();
    //   }
    // }
    std::vector<Edge *> outputs;
    return outputs;
  };
  virtual std::vector<Edge *> operator()(
      std::vector<Edge *> inputs) {
    this->markInputEdge(inputs);
    std::vector<Edge *> outputs = this->forward(inputs);
    if (graph_ != nullptr) {
      base::Status status = graph_->updateNodeIO(this, inputs, outputs);
      // for (auto input : inputs) {
      //   NNDEPLOY_LOGE("input->getName(): %s.\n", input->getName().c_str());
      // }
      // for (auto output : outputs) {
      //   NNDEPLOY_LOGE("output->getName(): %s.\n", output->getName().c_str());
      // }
      if (status != base::kStatusCodeOk) {
        NNDEPLOY_LOGE("graph_->updateNodeIO failed.\n");
        return std::vector<Edge *>();
      }
    }
    this->markOutputEdge(outputs);
    return outputs;
  }

  base::Status dump(std::ostream &oss = std::cout);

 protected:
  virtual base::Status construct();
  virtual base::Status executor();

 protected:
  bool is_graph_node_share_stream_ = true;
  std::vector<EdgeWrapper *> edge_repository_;
  std::vector<NodeWrapper *> node_repository_;
  std::vector<std::shared_ptr<Edge>> shared_edge_repository_;
  std::vector<std::shared_ptr<Node>> shared_node_repository_;
  std::set<std::string> used_node_names_;
  std::set<std::string> used_edge_names_;
  std::shared_ptr<Executor> executor_;
};

template <typename T, typename... Args,
          typename std::enable_if<std::is_base_of<Node, T>{}, int>::type>
Node *Graph::createNode(const std::string &name, Args &...args) {
  if (used_node_names_.find(name) != used_node_names_.end()) {
    NNDEPLOY_LOGE("node name[%s] is already used!\n", name.c_str());
    return nullptr;
  }
  std::vector<Edge *> inputs;
  std::vector<Edge *> outputs;
  Node *node = dynamic_cast<Node *>(new T(name, inputs, outputs, args...));
  NodeWrapper *node_wrapper = new NodeWrapper();
  node_wrapper->is_external_ = false;
  node_wrapper->node_ = node;
  node_wrapper->name_ = name;
  node_repository_.emplace_back(node_wrapper);
  used_node_names_.insert(name);
  node->setGraph(this);
  ;
  return node;
}

template <typename T, typename... Args,
          typename std::enable_if<std::is_base_of<Node, T>{}, int>::type>
Node *Graph::createNode(const std::string &name, Edge *input, Edge *output,
                        Args &...args) {
  if (used_node_names_.find(name) != used_node_names_.end()) {
    NNDEPLOY_LOGE("node name[%s] is already used!\n", name.c_str());
    return nullptr;
  }
  Node *node = dynamic_cast<Node *>(new T(name, {input}, {output}, args...));
  NodeWrapper *node_wrapper = new NodeWrapper();
  node_wrapper->is_external_ = false;
  node_wrapper->node_ = node;
  node_wrapper->name_ = name;
  EdgeWrapper *input_wrapper = findEdgeWrapper(edge_repository_, input);
  if (input_wrapper == nullptr) {
    input_wrapper = this->addEdge(input);
  }
  input_wrapper->consumers_.emplace_back(node_wrapper);
  EdgeWrapper *output_wrapper = findEdgeWrapper(edge_repository_, output);
  if (output_wrapper == nullptr) {
    output_wrapper = this->addEdge(output);
  }
  output_wrapper->producers_.emplace_back(node_wrapper);

  node_repository_.emplace_back(node_wrapper);
  used_node_names_.insert(name);
  node->setGraph(this);
  ;
  return node;
}

template <typename T, typename... Args,
          typename std::enable_if<std::is_base_of<Node, T>{}, int>::type>
Node *Graph::createNode(const std::string &name, const std::string &input_name,
                        const std::string &output_name, Args &...args) {
  if (used_node_names_.find(name) != used_node_names_.end()) {
    NNDEPLOY_LOGE("node name[%s] is already used!\n", name.c_str());
    return nullptr;
  }
  Edge *input = getEdge(input_name);
  if (input == nullptr) {
    input = createEdge(input_name);
  }
  Edge *output = getEdge(output_name);
  if (output == nullptr) {
    output = createEdge(output_name);
  }
  Node *node = dynamic_cast<Node *>(new T(name, {input}, {output}, args...));
  NodeWrapper *node_wrapper = new NodeWrapper();
  node_wrapper->is_external_ = false;
  node_wrapper->node_ = node;
  node_wrapper->name_ = name;
  EdgeWrapper *input_wrapper = findEdgeWrapper(edge_repository_, input);
  if (input_wrapper == nullptr) {
    input_wrapper = this->addEdge(input);
  }
  input_wrapper->consumers_.emplace_back(node_wrapper);
  EdgeWrapper *output_wrapper = findEdgeWrapper(edge_repository_, output);
  if (output_wrapper == nullptr) {
    output_wrapper = this->addEdge(output);
  }
  output_wrapper->producers_.emplace_back(node_wrapper);

  node_repository_.emplace_back(node_wrapper);
  used_node_names_.insert(name);
  node->setGraph(this);
  ;
  return node;
}

template <typename T, typename... Args,
          typename std::enable_if<std::is_base_of<Node, T>{}, int>::type>
Node *Graph::createNode(const std::string &name, Edge *input,
                        const std::string &output_name, Args &...args) {
  if (used_node_names_.find(name) != used_node_names_.end()) {
    NNDEPLOY_LOGE("node name[%s] is already used!\n", name.c_str());
    return nullptr;
  }
  Edge *output = getEdge(output_name);
  if (output == nullptr) {
    output = createEdge(output_name);
  }
  Node *node = dynamic_cast<Node *>(new T(name, {input}, {output}, args...));
  NodeWrapper *node_wrapper = new NodeWrapper();
  node_wrapper->is_external_ = false;
  node_wrapper->node_ = node;
  node_wrapper->name_ = name;
  EdgeWrapper *input_wrapper = findEdgeWrapper(edge_repository_, input);
  if (input_wrapper == nullptr) {
    input_wrapper = this->addEdge(input);
  }
  input_wrapper->consumers_.emplace_back(node_wrapper);
  EdgeWrapper *output_wrapper = findEdgeWrapper(edge_repository_, output);
  if (output_wrapper == nullptr) {
    output_wrapper = this->addEdge(output);
  }
  output_wrapper->producers_.emplace_back(node_wrapper);

  node_repository_.emplace_back(node_wrapper);
  used_node_names_.insert(name);
  node->setGraph(this);
  ;
  return node;
}

template <typename T, typename... Args,
          typename std::enable_if<std::is_base_of<Node, T>{}, int>::type>
Node *Graph::createNode(const std::string &name, const std::string &input_name,
                        Edge *output, Args &...args) {
  if (used_node_names_.find(name) != used_node_names_.end()) {
    NNDEPLOY_LOGE("node name[%s] is already used!\n", name.c_str());
    return nullptr;
  }
  Edge *input = getEdge(input_name);
  if (input == nullptr) {
    input = createEdge(input_name);
  }
  Node *node = dynamic_cast<Node *>(new T(name, {input}, {output}, args...));
  NodeWrapper *node_wrapper = new NodeWrapper();
  node_wrapper->is_external_ = false;
  node_wrapper->node_ = node;
  node_wrapper->name_ = name;
  EdgeWrapper *input_wrapper = findEdgeWrapper(edge_repository_, input);
  if (input_wrapper == nullptr) {
    input_wrapper = this->addEdge(input);
  }
  input_wrapper->consumers_.emplace_back(node_wrapper);
  EdgeWrapper *output_wrapper = findEdgeWrapper(edge_repository_, output);
  if (output_wrapper == nullptr) {
    output_wrapper = this->addEdge(output);
  }
  output_wrapper->producers_.emplace_back(node_wrapper);

  node_repository_.emplace_back(node_wrapper);
  used_node_names_.insert(name);
  node->setGraph(this);
  ;
  return node;
}

template <typename T, typename... Args,
          typename std::enable_if<std::is_base_of<Node, T>{}, int>::type>
Node *Graph::createNode(const std::string &name, std::vector<Edge *> inputs,
                        std::vector<Edge *> outputs, Args &...args) {
  if (used_node_names_.find(name) != used_node_names_.end()) {
    NNDEPLOY_LOGE("node name[%s] is already used!\n", name.c_str());
    return nullptr;
  }
  Node *node = dynamic_cast<Node *>(new T(name, inputs, outputs, args...));
  NodeWrapper *node_wrapper = new NodeWrapper();
  node_wrapper->is_external_ = false;
  node_wrapper->node_ = node;
  node_wrapper->name_ = name;
  for (auto input : inputs) {
    EdgeWrapper *input_wrapper = findEdgeWrapper(edge_repository_, input);
    if (input_wrapper == nullptr) {
      input_wrapper = this->addEdge(input);
    }
    input_wrapper->consumers_.emplace_back(node_wrapper);
  }
  for (auto output : outputs) {
    EdgeWrapper *output_wrapper = findEdgeWrapper(edge_repository_, output);
    if (output_wrapper == nullptr) {
      output_wrapper = this->addEdge(output);
    }
    output_wrapper->producers_.emplace_back(node_wrapper);
  }

  node_repository_.emplace_back(node_wrapper);
  used_node_names_.insert(name);
  node->setGraph(this);
  ;
  return node;
}

template <typename T, typename... Args,
          typename std::enable_if<std::is_base_of<Node, T>{}, int>::type>
Node *Graph::createNode(const std::string &name,
                        std::vector<std::string> input_names,
                        std::vector<std::string> output_names, Args &...args) {
  if (used_node_names_.find(name) != used_node_names_.end()) {
    NNDEPLOY_LOGE("node name[%s] is already used!\n", name.c_str());
    return nullptr;
  }
  std::vector<Edge *> inputs;
  for (auto input_name : input_names) {
    Edge *input = getEdge(input_name);
    if (input == nullptr) {
      input = createEdge(input_name);
    }
    inputs.emplace_back(input);
  }
  std::vector<Edge *> outputs;
  for (auto output_name : output_names) {
    Edge *output = getEdge(output_name);
    if (output == nullptr) {
      output = createEdge(output_name);
    }
    outputs.emplace_back(output);
  }
  Node *node = dynamic_cast<Node *>(new T(name, inputs, outputs, args...));
  NodeWrapper *node_wrapper = new NodeWrapper();
  node_wrapper->is_external_ = false;
  node_wrapper->node_ = node;
  node_wrapper->name_ = name;
  for (auto input : inputs) {
    EdgeWrapper *input_wrapper = findEdgeWrapper(edge_repository_, input);
    if (input_wrapper == nullptr) {
      input_wrapper = this->addEdge(input);
    }
    input_wrapper->consumers_.emplace_back(node_wrapper);
  }
  for (auto output : outputs) {
    EdgeWrapper *output_wrapper = findEdgeWrapper(edge_repository_, output);
    if (output_wrapper == nullptr) {
      output_wrapper = this->addEdge(output);
    }
    output_wrapper->producers_.emplace_back(node_wrapper);
  }

  node_repository_.emplace_back(node_wrapper);
  used_node_names_.insert(name);
  node->setGraph(this);
  ;
  return node;
}

template <typename T, typename... Args,
          typename std::enable_if<std::is_base_of<Node, T>{}, int>::type>
Node *Graph::createNode(const std::string &name,
                        std::vector<std::string> input_names,
                        std::vector<Edge *> outputs, Args &...args) {
  if (used_node_names_.find(name) != used_node_names_.end()) {
    NNDEPLOY_LOGE("node name[%s] is already used!\n", name.c_str());
    return nullptr;
  }
  std::vector<Edge *> inputs;
  for (auto input_name : input_names) {
    Edge *input = getEdge(input_name);
    if (input == nullptr) {
      input = createEdge(input_name);
    }
    inputs.emplace_back(input);
  }
  Node *node = dynamic_cast<Node *>(new T(name, inputs, outputs, args...));
  NodeWrapper *node_wrapper = new NodeWrapper();
  node_wrapper->is_external_ = false;
  node_wrapper->node_ = node;
  node_wrapper->name_ = name;
  for (auto input : inputs) {
    EdgeWrapper *input_wrapper = findEdgeWrapper(edge_repository_, input);
    if (input_wrapper == nullptr) {
      input_wrapper = this->addEdge(input);
    }
    input_wrapper->consumers_.emplace_back(node_wrapper);
  }
  for (auto output : outputs) {
    EdgeWrapper *output_wrapper = findEdgeWrapper(edge_repository_, output);
    if (output_wrapper == nullptr) {
      output_wrapper = this->addEdge(output);
    }
    output_wrapper->producers_.emplace_back(node_wrapper);
  }

  node_repository_.emplace_back(node_wrapper);
  used_node_names_.insert(name);
  node->setGraph(this);
  ;
  return node;
}

template <typename T, typename... Args,
          typename std::enable_if<std::is_base_of<Node, T>{}, int>::type>
Node *Graph::createNode(const std::string &name, std::vector<Edge *> inputs,
                        std::vector<std::string> output_names, Args &...args) {
  if (used_node_names_.find(name) != used_node_names_.end()) {
    NNDEPLOY_LOGE("node name[%s] is already used!\n", name.c_str());
    return nullptr;
  }
  std::vector<Edge *> outputs;
  for (auto output_name : output_names) {
    Edge *output = getEdge(output_name);
    if (output == nullptr) {
      output = createEdge(output_name);
    }
    outputs.emplace_back(output);
  }
  Node *node = dynamic_cast<Node *>(new T(name, inputs, outputs, args...));
  NodeWrapper *node_wrapper = new NodeWrapper();
  node_wrapper->is_external_ = false;
  node_wrapper->node_ = node;
  node_wrapper->name_ = name;
  for (auto input : inputs) {
    EdgeWrapper *input_wrapper = findEdgeWrapper(edge_repository_, input);
    if (input_wrapper == nullptr) {
      input_wrapper = this->addEdge(input);
    }
    input_wrapper->consumers_.emplace_back(node_wrapper);
  }
  for (auto output : outputs) {
    EdgeWrapper *output_wrapper = findEdgeWrapper(edge_repository_, output);
    if (output_wrapper == nullptr) {
      output_wrapper = this->addEdge(output);
    }
    output_wrapper->producers_.emplace_back(node_wrapper);
  }

  node_repository_.emplace_back(node_wrapper);
  used_node_names_.insert(name);
  node->setGraph(this);
  ;
  return node;
}

template <typename T, typename... Args,
          typename std::enable_if<std::is_base_of<Node, T>{}, int>::type>
Node *Graph::createNode(const std::string &name,
                        std::initializer_list<Edge *> inputs,
                        std::initializer_list<Edge *> outputs, Args &...args) {
  if (used_node_names_.find(name) != used_node_names_.end()) {
    NNDEPLOY_LOGE("node name[%s] is already used!\n", name.c_str());
    return nullptr;
  }
  Node *node = dynamic_cast<Node *>(new T(name, inputs, outputs, args...));
  NodeWrapper *node_wrapper = new NodeWrapper();
  node_wrapper->is_external_ = false;
  node_wrapper->node_ = node;
  node_wrapper->name_ = name;
  for (auto input : inputs) {
    EdgeWrapper *input_wrapper = findEdgeWrapper(edge_repository_, input);
    if (input_wrapper == nullptr) {
      input_wrapper = this->addEdge(input);
    }
    input_wrapper->consumers_.emplace_back(node_wrapper);
  }
  for (auto output : outputs) {
    EdgeWrapper *output_wrapper = findEdgeWrapper(edge_repository_, output);
    if (output_wrapper == nullptr) {
      output_wrapper = this->addEdge(output);
    }
    output_wrapper->producers_.emplace_back(node_wrapper);
  }

  node_repository_.emplace_back(node_wrapper);
  used_node_names_.insert(name);
  node->setGraph(this);
  ;
  return node;
}

template <typename T, typename... Args,
          typename std::enable_if<std::is_base_of<Node, T>{}, int>::type>
Node *Graph::createNode(const std::string &name,
                        std::initializer_list<std::string> input_names,
                        std::initializer_list<std::string> output_names,
                        Args &...args) {
  if (used_node_names_.find(name) != used_node_names_.end()) {
    NNDEPLOY_LOGE("node name[%s] is already used!\n", name.c_str());
    return nullptr;
  }
  std::vector<Edge *> inputs;
  for (auto input_name : input_names) {
    Edge *input = getEdge(input_name);
    if (input == nullptr) {
      input = createEdge(input_name);
    }
    inputs.emplace_back(input);
  }
  std::vector<Edge *> outputs;
  for (auto output_name : output_names) {
    Edge *output = getEdge(output_name);
    if (output == nullptr) {
      output = createEdge(output_name);
    }
    outputs.emplace_back(output);
  }
  Node *node = dynamic_cast<Node *>(new T(name, inputs, outputs, args...));
  NodeWrapper *node_wrapper = new NodeWrapper();
  node_wrapper->is_external_ = false;
  node_wrapper->node_ = node;
  node_wrapper->name_ = name;
  for (auto input : inputs) {
    EdgeWrapper *input_wrapper = findEdgeWrapper(edge_repository_, input);
    if (input_wrapper == nullptr) {
      input_wrapper = this->addEdge(input);
    }
    input_wrapper->consumers_.emplace_back(node_wrapper);
  }
  for (auto output : outputs) {
    EdgeWrapper *output_wrapper = findEdgeWrapper(edge_repository_, output);
    if (output_wrapper == nullptr) {
      output_wrapper = this->addEdge(output);
    }
    output_wrapper->producers_.emplace_back(node_wrapper);
  }

  node_repository_.emplace_back(node_wrapper);
  used_node_names_.insert(name);
  node->setGraph(this);
  ;
  return node;
}

template <typename T, typename... Args,
          typename std::enable_if<std::is_base_of<Node, T>{}, int>::type>
Node *Graph::createNode(const std::string &name,
                        std::initializer_list<Edge *> inputs,
                        std::initializer_list<std::string> output_names,
                        Args &...args) {
  if (used_node_names_.find(name) != used_node_names_.end()) {
    NNDEPLOY_LOGE("node name[%s] is already used!\n", name.c_str());
    return nullptr;
  }
  std::vector<Edge *> outputs;
  for (auto output_name : output_names) {
    Edge *output = getEdge(output_name);
    if (output == nullptr) {
      output = createEdge(output_name);
    }
    outputs.emplace_back(output);
  }
  Node *node = dynamic_cast<Node *>(new T(name, inputs, outputs, args...));
  NodeWrapper *node_wrapper = new NodeWrapper();
  node_wrapper->is_external_ = false;
  node_wrapper->node_ = node;
  node_wrapper->name_ = name;
  for (auto input : inputs) {
    EdgeWrapper *input_wrapper = findEdgeWrapper(edge_repository_, input);
    if (input_wrapper == nullptr) {
      input_wrapper = this->addEdge(input);
    }
    input_wrapper->consumers_.emplace_back(node_wrapper);
  }
  for (auto output : outputs) {
    EdgeWrapper *output_wrapper = findEdgeWrapper(edge_repository_, output);
    if (output_wrapper == nullptr) {
      output_wrapper = this->addEdge(output);
    }
    output_wrapper->producers_.emplace_back(node_wrapper);
  }

  node_repository_.emplace_back(node_wrapper);
  used_node_names_.insert(name);
  node->setGraph(this);
  ;
  return node;
}

template <typename T, typename... Args,
          typename std::enable_if<std::is_base_of<Node, T>{}, int>::type>
Node *Graph::createNode(const std::string &name,
                        std::initializer_list<std::string> input_names,
                        std::initializer_list<Edge *> outputs, Args &...args) {
  if (used_node_names_.find(name) != used_node_names_.end()) {
    NNDEPLOY_LOGE("node name[%s] is already used!\n", name.c_str());
    return nullptr;
  }
  std::vector<Edge *> inputs;
  for (auto input_name : input_names) {
    Edge *input = getEdge(input_name);
    if (input == nullptr) {
      input = createEdge(input_name);
    }
    inputs.emplace_back(input);
  }
  Node *node = dynamic_cast<Node *>(new T(name, inputs, outputs, args...));
  NodeWrapper *node_wrapper = new NodeWrapper();
  node_wrapper->is_external_ = false;
  node_wrapper->node_ = node;
  node_wrapper->name_ = name;
  for (auto input : inputs) {
    EdgeWrapper *input_wrapper = findEdgeWrapper(edge_repository_, input);
    if (input_wrapper == nullptr) {
      input_wrapper = this->addEdge(input);
    }
    input_wrapper->consumers_.emplace_back(node_wrapper);
  }
  for (auto output : outputs) {
    EdgeWrapper *output_wrapper = findEdgeWrapper(edge_repository_, output);
    if (output_wrapper == nullptr) {
      output_wrapper = this->addEdge(output);
    }
    output_wrapper->producers_.emplace_back(node_wrapper);
  }

  node_repository_.emplace_back(node_wrapper);
  used_node_names_.insert(name);
  node->setGraph(this);
  ;
  return node;
}

template <typename T, typename... Args,
          typename std::enable_if<std::is_base_of<Node, T>{}, int>::type>
Node *Graph::createInfer(const std::string &name, base::InferenceType type,
                         Edge *input, Edge *output) {
  if (used_node_names_.find(name) != used_node_names_.end()) {
    NNDEPLOY_LOGE("node name[%s] is already used!\n", name.c_str());
    return nullptr;
  }
  Node *node = dynamic_cast<Node *>(new T(name, {input}, {output}, type));
  NodeWrapper *node_wrapper = new NodeWrapper();
  node_wrapper->is_external_ = false;
  node_wrapper->node_ = node;
  node_wrapper->name_ = name;
  EdgeWrapper *input_wrapper = findEdgeWrapper(edge_repository_, input);
  if (input_wrapper == nullptr) {
    input_wrapper = this->addEdge(input);
  }
  input_wrapper->consumers_.emplace_back(node_wrapper);
  EdgeWrapper *output_wrapper = findEdgeWrapper(edge_repository_, output);
  if (output_wrapper == nullptr) {
    output_wrapper = this->addEdge(output);
  }
  output_wrapper->producers_.emplace_back(node_wrapper);

  node_repository_.emplace_back(node_wrapper);
  used_node_names_.insert(name);
  node->setGraph(this);
  ;
  return node;
}

template <typename T, typename... Args,
          typename std::enable_if<std::is_base_of<Node, T>{}, int>::type>
Node *Graph::createInfer(const std::string &name, base::InferenceType type,
                         const std::string &input_name,
                         const std::string &output_name) {
  if (used_node_names_.find(name) != used_node_names_.end()) {
    NNDEPLOY_LOGE("node name[%s] is already used!\n", name.c_str());
    return nullptr;
  }
  Edge *input = getEdge(input_name);
  if (input == nullptr) {
    input = createEdge(input_name);
  }
  Edge *output = getEdge(output_name);
  if (output == nullptr) {
    output = createEdge(output_name);
  }
  Node *node = dynamic_cast<Node *>(new T(name, {input}, {output}, type));
  NodeWrapper *node_wrapper = new NodeWrapper();
  node_wrapper->is_external_ = false;
  node_wrapper->node_ = node;
  node_wrapper->name_ = name;
  EdgeWrapper *input_wrapper = findEdgeWrapper(edge_repository_, input);
  if (input_wrapper == nullptr) {
    input_wrapper = this->addEdge(input);
  }
  input_wrapper->consumers_.emplace_back(node_wrapper);
  EdgeWrapper *output_wrapper = findEdgeWrapper(edge_repository_, output);
  if (output_wrapper == nullptr) {
    output_wrapper = this->addEdge(output);
  }
  output_wrapper->producers_.emplace_back(node_wrapper);

  node_repository_.emplace_back(node_wrapper);
  used_node_names_.insert(name);
  node->setGraph(this);
  ;
  return node;
}

template <typename T, typename... Args,
          typename std::enable_if<std::is_base_of<Node, T>{}, int>::type>
Node *Graph::createInfer(const std::string &name, base::InferenceType type,
                         Edge *input, const std::string &output_name) {
  if (used_node_names_.find(name) != used_node_names_.end()) {
    NNDEPLOY_LOGE("node name[%s] is already used!\n", name.c_str());
    return nullptr;
  }
  Edge *output = getEdge(output_name);
  if (output == nullptr) {
    output = createEdge(output_name);
  }
  Node *node = dynamic_cast<Node *>(new T(name, {input}, {output}, type));
  NodeWrapper *node_wrapper = new NodeWrapper();
  node_wrapper->is_external_ = false;
  node_wrapper->node_ = node;
  node_wrapper->name_ = name;
  EdgeWrapper *input_wrapper = findEdgeWrapper(edge_repository_, input);
  if (input_wrapper == nullptr) {
    input_wrapper = this->addEdge(input);
  }
  input_wrapper->consumers_.emplace_back(node_wrapper);
  EdgeWrapper *output_wrapper = findEdgeWrapper(edge_repository_, output);
  if (output_wrapper == nullptr) {
    output_wrapper = this->addEdge(output);
  }
  output_wrapper->producers_.emplace_back(node_wrapper);

  node_repository_.emplace_back(node_wrapper);
  used_node_names_.insert(name);
  node->setGraph(this);
  ;
  return node;
}

template <typename T, typename... Args,
          typename std::enable_if<std::is_base_of<Node, T>{}, int>::type>
Node *Graph::createInfer(const std::string &name, base::InferenceType type,
                         const std::string &input_name, Edge *output) {
  if (used_node_names_.find(name) != used_node_names_.end()) {
    NNDEPLOY_LOGE("node name[%s] is already used!\n", name.c_str());
    return nullptr;
  }
  Edge *input = getEdge(input_name);
  if (input == nullptr) {
    input = createEdge(input_name);
  }
  Node *node = dynamic_cast<Node *>(new T(name, {input}, {output}, type));
  NodeWrapper *node_wrapper = new NodeWrapper();
  node_wrapper->is_external_ = false;
  node_wrapper->node_ = node;
  node_wrapper->name_ = name;
  EdgeWrapper *input_wrapper = findEdgeWrapper(edge_repository_, input);
  if (input_wrapper == nullptr) {
    input_wrapper = this->addEdge(input);
  }
  input_wrapper->consumers_.emplace_back(node_wrapper);
  EdgeWrapper *output_wrapper = findEdgeWrapper(edge_repository_, output);
  if (output_wrapper == nullptr) {
    output_wrapper = this->addEdge(output);
  }
  output_wrapper->producers_.emplace_back(node_wrapper);

  node_repository_.emplace_back(node_wrapper);
  used_node_names_.insert(name);
  node->setGraph(this);
  ;
  return node;
}

template <typename T, typename... Args,
          typename std::enable_if<std::is_base_of<Node, T>{}, int>::type>
Node *Graph::createInfer(const std::string &name, base::InferenceType type,
                         std::vector<Edge *> inputs,
                         std::vector<Edge *> outputs) {
  if (used_node_names_.find(name) != used_node_names_.end()) {
    NNDEPLOY_LOGE("node name[%s] is already used!\n", name.c_str());
    return nullptr;
  }
  Node *node = dynamic_cast<Node *>(new T(name, inputs, outputs, type));
  NodeWrapper *node_wrapper = new NodeWrapper();
  node_wrapper->is_external_ = false;
  node_wrapper->node_ = node;
  node_wrapper->name_ = name;
  for (auto input : inputs) {
    EdgeWrapper *input_wrapper = findEdgeWrapper(edge_repository_, input);
    if (input_wrapper == nullptr) {
      input_wrapper = this->addEdge(input);
    }
    input_wrapper->consumers_.emplace_back(node_wrapper);
  }
  for (auto output : outputs) {
    EdgeWrapper *output_wrapper = findEdgeWrapper(edge_repository_, output);
    if (output_wrapper == nullptr) {
      output_wrapper = this->addEdge(output);
    }
    output_wrapper->producers_.emplace_back(node_wrapper);
  }

  node_repository_.emplace_back(node_wrapper);
  used_edge_names_.insert(name);
  node->setGraph(this);
  ;
  return node;
}

template <typename T, typename... Args,
          typename std::enable_if<std::is_base_of<Node, T>{}, int>::type>
Node *Graph::createInfer(const std::string &name, base::InferenceType type,
                         std::vector<std::string> input_names,
                         std::vector<std::string> output_names) {
  if (used_node_names_.find(name) != used_node_names_.end()) {
    NNDEPLOY_LOGE("node name[%s] is already used!\n", name.c_str());
    return nullptr;
  }
  std::vector<Edge *> inputs;
  for (auto input_name : input_names) {
    Edge *input = getEdge(input_name);
    if (input == nullptr) {
      input = createEdge(input_name);
    }
    inputs.emplace_back(input);
  }
  std::vector<Edge *> outputs;
  for (auto output_name : output_names) {
    Edge *output = getEdge(output_name);
    if (output == nullptr) {
      output = createEdge(output_name);
    }
    outputs.emplace_back(output);
  }
  Node *node = dynamic_cast<Node *>(new T(name, inputs, outputs, type));
  NodeWrapper *node_wrapper = new NodeWrapper();
  node_wrapper->is_external_ = false;
  node_wrapper->node_ = node;
  node_wrapper->name_ = name;
  for (auto input : inputs) {
    EdgeWrapper *input_wrapper = findEdgeWrapper(edge_repository_, input);
    if (input_wrapper == nullptr) {
      input_wrapper = this->addEdge(input);
    }
    input_wrapper->consumers_.emplace_back(node_wrapper);
  }
  for (auto output : outputs) {
    EdgeWrapper *output_wrapper = findEdgeWrapper(edge_repository_, output);
    if (output_wrapper == nullptr) {
      output_wrapper = this->addEdge(output);
    }
    output_wrapper->producers_.emplace_back(node_wrapper);
  }

  node_repository_.emplace_back(node_wrapper);
  used_node_names_.insert(name);
  node->setGraph(this);
  ;
  return node;
}

template <typename T, typename... Args,
          typename std::enable_if<std::is_base_of<Node, T>{}, int>::type>
Node *Graph::createInfer(const std::string &name, base::InferenceType type,
                         std::vector<Edge *> inputs,
                         std::vector<std::string> output_names) {
  if (used_node_names_.find(name) != used_node_names_.end()) {
    NNDEPLOY_LOGE("node name[%s] is already used!\n", name.c_str());
    return nullptr;
  }
  std::vector<Edge *> outputs;
  for (auto output_name : output_names) {
    Edge *output = getEdge(output_name);
    if (output == nullptr) {
      output = createEdge(output_name);
    }
    outputs.emplace_back(output);
  }
  Node *node = dynamic_cast<Node *>(new T(name, inputs, outputs, type));
  NodeWrapper *node_wrapper = new NodeWrapper();
  node_wrapper->is_external_ = false;
  node_wrapper->node_ = node;
  node_wrapper->name_ = name;
  for (auto input : inputs) {
    EdgeWrapper *input_wrapper = findEdgeWrapper(edge_repository_, input);
    if (input_wrapper == nullptr) {
      input_wrapper = this->addEdge(input);
    }
    input_wrapper->consumers_.emplace_back(node_wrapper);
  }
  for (auto output : outputs) {
    EdgeWrapper *output_wrapper = findEdgeWrapper(edge_repository_, output);
    if (output_wrapper == nullptr) {
      output_wrapper = this->addEdge(output);
    }
    output_wrapper->producers_.emplace_back(node_wrapper);
  }

  node_repository_.emplace_back(node_wrapper);
  used_node_names_.insert(name);
  node->setGraph(this);
  ;
  return node;
}

template <typename T, typename... Args,
          typename std::enable_if<std::is_base_of<Node, T>{}, int>::type>
Node *Graph::createInfer(const std::string &name, base::InferenceType type,
                         std::vector<std::string> input_names,
                         std::vector<Edge *> outputs) {
  if (used_node_names_.find(name) != used_node_names_.end()) {
    NNDEPLOY_LOGE("node name[%s] is already used!\n", name.c_str());
    return nullptr;
  }
  std::vector<Edge *> inputs;
  for (auto input_name : input_names) {
    Edge *input = getEdge(input_name);
    if (input == nullptr) {
      input = createEdge(input_name);
    }
    inputs.emplace_back(input);
  }
  Node *node = dynamic_cast<Node *>(new T(name, inputs, outputs, type));
  NodeWrapper *node_wrapper = new NodeWrapper();
  node_wrapper->is_external_ = false;
  node_wrapper->node_ = node;
  node_wrapper->name_ = name;
  for (auto input : inputs) {
    EdgeWrapper *input_wrapper = findEdgeWrapper(edge_repository_, input);
    if (input_wrapper == nullptr) {
      input_wrapper = this->addEdge(input);
    }
    input_wrapper->consumers_.emplace_back(node_wrapper);
  }
  for (auto output : outputs) {
    EdgeWrapper *output_wrapper = findEdgeWrapper(edge_repository_, output);
    if (output_wrapper == nullptr) {
      output_wrapper = this->addEdge(output);
    }
    output_wrapper->producers_.emplace_back(node_wrapper);
  }

  node_repository_.emplace_back(node_wrapper);
  used_node_names_.insert(name);
  node->setGraph(this);
  ;
  return node;
}

template <typename T, typename... Args,
          typename std::enable_if<std::is_base_of<Node, T>{}, int>::type>
Node *Graph::createInfer(const std::string &name, base::InferenceType type,
                         std::initializer_list<Edge *> inputs,
                         std::initializer_list<Edge *> outputs) {
  if (used_node_names_.find(name) != used_node_names_.end()) {
    NNDEPLOY_LOGE("node name[%s] is already used!\n", name.c_str());
    return nullptr;
  }
  Node *node = dynamic_cast<Node *>(new T(name, inputs, outputs, type));
  NodeWrapper *node_wrapper = new NodeWrapper();
  node_wrapper->is_external_ = false;
  node_wrapper->node_ = node;
  node_wrapper->name_ = name;
  for (auto input : inputs) {
    EdgeWrapper *input_wrapper = findEdgeWrapper(edge_repository_, input);
    if (input_wrapper == nullptr) {
      input_wrapper = this->addEdge(input);
    }
    input_wrapper->consumers_.emplace_back(node_wrapper);
  }
  for (auto output : outputs) {
    EdgeWrapper *output_wrapper = findEdgeWrapper(edge_repository_, output);
    if (output_wrapper == nullptr) {
      output_wrapper = this->addEdge(output);
    }
    output_wrapper->producers_.emplace_back(node_wrapper);
  }

  node_repository_.emplace_back(node_wrapper);
  used_node_names_.insert(name);
  node->setGraph(this);
  ;
  return node;
}

template <typename T, typename... Args,
          typename std::enable_if<std::is_base_of<Node, T>{}, int>::type>
Node *Graph::createInfer(const std::string &name, base::InferenceType type,
                         std::initializer_list<std::string> input_names,
                         std::initializer_list<std::string> output_names) {
  if (used_node_names_.find(name) != used_node_names_.end()) {
    NNDEPLOY_LOGE("node name[%s] is already used!\n", name.c_str());
    return nullptr;
  }
  std::vector<Edge *> inputs;
  for (auto input_name : input_names) {
    Edge *input = getEdge(input_name);
    if (input == nullptr) {
      input = createEdge(input_name);
    }
    inputs.emplace_back(input);
  }
  std::vector<Edge *> outputs;
  for (auto output_name : output_names) {
    Edge *output = getEdge(output_name);
    if (output == nullptr) {
      output = createEdge(output_name);
    }
    outputs.emplace_back(output);
  }
  Node *node = dynamic_cast<Node *>(new T(name, inputs, outputs, type));
  NodeWrapper *node_wrapper = new NodeWrapper();
  node_wrapper->is_external_ = false;
  node_wrapper->node_ = node;
  node_wrapper->name_ = name;
  for (auto input : inputs) {
    EdgeWrapper *input_wrapper = findEdgeWrapper(edge_repository_, input);
    if (input_wrapper == nullptr) {
      input_wrapper = this->addEdge(input);
    }
    input_wrapper->consumers_.emplace_back(node_wrapper);
  }
  for (auto output : outputs) {
    EdgeWrapper *output_wrapper = findEdgeWrapper(edge_repository_, output);
    if (output_wrapper == nullptr) {
      output_wrapper = this->addEdge(output);
    }
    output_wrapper->producers_.emplace_back(node_wrapper);
  }

  node_repository_.emplace_back(node_wrapper);
  used_node_names_.insert(name);
  node->setGraph(this);
  ;
  return node;
}

template <typename T, typename... Args,
          typename std::enable_if<std::is_base_of<Node, T>{}, int>::type>
Node *Graph::createInfer(const std::string &name, base::InferenceType type,
                         std::initializer_list<Edge *> inputs,
                         std::initializer_list<std::string> output_names) {
  if (used_node_names_.find(name) != used_node_names_.end()) {
    NNDEPLOY_LOGE("node name[%s] is already used!\n", name.c_str());
    return nullptr;
  }
  std::vector<Edge *> outputs;
  for (auto output_name : output_names) {
    Edge *output = getEdge(output_name);
    if (output == nullptr) {
      output = createEdge(output_name);
    }
    outputs.emplace_back(output);
  }
  Node *node = dynamic_cast<Node *>(new T(name, inputs, outputs, type));
  NodeWrapper *node_wrapper = new NodeWrapper();
  node_wrapper->is_external_ = false;
  node_wrapper->node_ = node;
  node_wrapper->name_ = name;
  for (auto input : inputs) {
    EdgeWrapper *input_wrapper = findEdgeWrapper(edge_repository_, input);
    if (input_wrapper == nullptr) {
      input_wrapper = this->addEdge(input);
    }
    input_wrapper->consumers_.emplace_back(node_wrapper);
  }
  for (auto output : outputs) {
    EdgeWrapper *output_wrapper = findEdgeWrapper(edge_repository_, output);
    if (output_wrapper == nullptr) {
      output_wrapper = this->addEdge(output);
    }
    output_wrapper->producers_.emplace_back(node_wrapper);
  }

  node_repository_.emplace_back(node_wrapper);
  used_node_names_.insert(name);
  node->setGraph(this);
  ;
  return node;
}

template <typename T, typename... Args,
          typename std::enable_if<std::is_base_of<Node, T>{}, int>::type>
Node *Graph::createInfer(const std::string &name, base::InferenceType type,
                         std::initializer_list<std::string> input_names,
                         std::initializer_list<Edge *> outputs) {
  if (used_node_names_.find(name) != used_node_names_.end()) {
    NNDEPLOY_LOGE("node name[%s] is already used!\n", name.c_str());
    return nullptr;
  }
  std::vector<Edge *> inputs;
  for (auto input_name : input_names) {
    Edge *input = getEdge(input_name);
    if (input == nullptr) {
      input = createEdge(input_name);
    }
    inputs.emplace_back(input);
  }
  std::vector<Edge *> outputs_vec;
  for (auto output : outputs) {
    outputs_vec.emplace_back(output);
  }
  Node *node = dynamic_cast<Node *>(new T(name, inputs, outputs_vec, type));
  NodeWrapper *node_wrapper = new NodeWrapper();
  node_wrapper->is_external_ = false;
  node_wrapper->node_ = node;
  node_wrapper->name_ = name;
  for (auto input : inputs) {
    EdgeWrapper *input_wrapper = findEdgeWrapper(edge_repository_, input);
    if (input_wrapper == nullptr) {
      input_wrapper = this->addEdge(input);
    }
    input_wrapper->consumers_.emplace_back(node_wrapper);
  }
  for (auto output : outputs) {
    EdgeWrapper *output_wrapper = findEdgeWrapper(edge_repository_, output);
    if (output_wrapper == nullptr) {
      output_wrapper = this->addEdge(output);
    }
    output_wrapper->producers_.emplace_back(node_wrapper);
  }

  node_repository_.emplace_back(node_wrapper);
  used_node_names_.insert(name);
  node->setGraph(this);
  ;
  return node;
}

// template <typename T, typename... Args,
//           typename std::enable_if<std::is_base_of<Node, T>{}, int>::type>
// Node *Graph::createNodeSiso(const NodeDesc &desc) {
//   Node *node = nullptr;
//   std::vector<std::string> inputs = desc.getInputs();
//   std::vector<std::string> outputs = desc.getOutputs();
//   if (inputs.size() != 1 || outputs.size() != 1) {
//     NNDEPLOY_LOGE("node desc is invalid!\n");
//     return node;
//   }
//   node = this->createNode<T>(desc.getName(), inputs[0], outputs[0]);
//   if (node == nullptr) {
//     NNDEPLOY_LOGE("create node[%s] failed!\n", desc.getName().c_str());
//     return node;
//   }
//   std::shared_ptr<base::Param> param = desc.getParam();
//   if (param != nullptr) {
//     node->setParam(param.get());
//   }
//   return node;
// }
// template <typename T, typename... Args,
//           typename std::enable_if<std::is_base_of<Node, T>{}, int>::type>
// Node *Graph::createNodeMimo(const NodeDesc &desc) {
//   Node *node = nullptr;
//   std::vector<std::string> inputs = desc.getInputs();
//   std::vector<std::string> outputs = desc.getOutputs();
//   node = this->createNode<T>(desc.getName(), inputs, outputs);
//   if (node == nullptr) {
//     NNDEPLOY_LOGE("create node[%s] failed!\n", desc.getName().c_str());
//     return node;
//   }
//   std::shared_ptr<base::Param> param = desc.getParam();
//   if (param != nullptr) {
//     node->setParam(param.get());
//   }
//   return node;
// }

template <typename T, typename... Args,
          typename std::enable_if<std::is_base_of<Node, T>{}, int>::type>
Node *Graph::createNode(const NodeDesc &desc, Args &...args) {
  Node *node = this->createNode<T>(desc.getName(), desc.getInputs(),
                                   desc.getOutputs(), args...);
  if (node == nullptr) {
    NNDEPLOY_LOGE("create infer node[%s] failed!\n", desc.getName().c_str());
    return node;
  }
  node->setGraph(this);
  return node;
}

using createGraphFunc = std::function<Graph *(
    const std::string &name, base::InferenceType inference_type,
    base::DeviceType device_type, Edge *input, Edge *output,
    base::ModelType model_type, bool is_path,
    std::vector<std::string> model_value)>;

std::map<std::string, createGraphFunc> &getGlobalGraphCreatorMap();

class TypeGraphRegister {
 public:
  explicit TypeGraphRegister(const std::string &name, createGraphFunc func) {
    getGlobalGraphCreatorMap()[name] = func;
  }
};

extern NNDEPLOY_CC_API Graph *createGraph(const std::string &name,
                                          base::InferenceType inference_type,
                                          base::DeviceType device_type,
                                          Edge *input, Edge *output,
                                          base::ModelType model_type,
                                          bool is_path,
                                          std::vector<std::string> model_value);

}  // namespace dag
}  // namespace nndeploy

#endif  // _NNDEPLOY_DAG_EXECUTOR_H_
