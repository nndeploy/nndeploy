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
  Graph(const std::string &name, Edge *input, Edge *output);
  Graph(const std::string &name, std::initializer_list<Edge *> inputs,
        std::initializer_list<Edge *> outputs);
  Graph(const std::string &name, std::vector<Edge *> inputs,
        std::vector<Edge *> outputs);
  virtual ~Graph();

  /**
   * @brief 在Graph中创建一条Edge
   * @param  name             edge名称
   * @return Edge*
   */
  Edge *createEdge(const std::string &name);

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

  /**
   * @brief 获取Graph中的Edge
   * @param  name             edge名称
   * @return Edge*
   */
  Edge *getEdge(const std::string &name);

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

  // base::Status addNode(Node *node);

  /**
   * @brief 将一个已有的Node加入Graph
   * @param  node             已有Node
   * @return base::Status
   */
  base::Status addNode(Node *node, bool is_external = true);

  base::Status setNodeParam(const std::string &node_name, base::Param *param);
  base::Param *getNodeParam(const std::string &node_name);

  virtual base::Status init();
  virtual base::Status deinit();

  virtual base::Status run();

  base::Status dump(std::ostream &oss = std::cout);

 protected:
  virtual base::Status construct();
  virtual base::Status executor();

 protected:
  std::vector<EdgeWrapper *> edge_repository_;
  std::vector<NodeWrapper *> node_repository_;
  std::set<std::string> used_node_names_;
  std::set<std::string> used_edge_names_;
  std::shared_ptr<Executor> executor_;
};

template <typename T, typename... Args,
          typename std::enable_if<std::is_base_of<Node, T>{}, int>::type>
Node *Graph::createNode(const std::string &name, Edge *input, Edge *output,
                        Args &...args) {
  if (used_node_names_.find(name) != used_node_names_.end()){
    NNDEPLOY_LOGE("node name[%s] is already used!\n", name.c_str());
    return nullptr;
  }
  Node *node = dynamic_cast<Node *>(new T(name, input, output, args...));
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
  return node;
}

template <typename T, typename... Args,
          typename std::enable_if<std::is_base_of<Node, T>{}, int>::type>
Node *Graph::createNode(const std::string &name, const std::string &input_name,
                        const std::string &output_name, Args &...args) {
  if (used_node_names_.find(name) != used_node_names_.end()){
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
  Node *node = dynamic_cast<Node *>(new T(name, input, output, args...));
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
  return node;
}

template <typename T, typename... Args,
          typename std::enable_if<std::is_base_of<Node, T>{}, int>::type>
Node *Graph::createNode(const std::string &name, Edge *input,
                        const std::string &output_name, Args &...args) {
  if (used_node_names_.find(name) != used_node_names_.end()){
    NNDEPLOY_LOGE("node name[%s] is already used!\n", name.c_str());
    return nullptr;
  }
  Edge *output = getEdge(output_name);
  if (output == nullptr) {
    output = createEdge(output_name);
  }
  Node *node = dynamic_cast<Node *>(new T(name, input, output, args...));
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
  return node;
}

template <typename T, typename... Args,
          typename std::enable_if<std::is_base_of<Node, T>{}, int>::type>
Node *Graph::createNode(const std::string &name, const std::string &input_name,
                        Edge *output, Args &...args) {
    if (used_node_names_.find(name) != used_node_names_.end()){
      NNDEPLOY_LOGE("node name[%s] is already used!\n", name.c_str());
    return nullptr;
  }
  Edge *input = getEdge(input_name);
  if (input == nullptr) {
    input = createEdge(input_name);
  }
  Node *node = dynamic_cast<Node *>(new T(name, input, output, args...));
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
  return node;
}

template <typename T, typename... Args,
          typename std::enable_if<std::is_base_of<Node, T>{}, int>::type>
Node *Graph::createNode(const std::string &name, std::vector<Edge *> inputs,
                        std::vector<Edge *> outputs, Args &...args) {
  if (used_node_names_.find(name) != used_node_names_.end()){
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
  return node;
}

template <typename T, typename... Args,
          typename std::enable_if<std::is_base_of<Node, T>{}, int>::type>
Node *Graph::createNode(const std::string &name,
                        std::vector<std::string> input_names,
                        std::vector<std::string> output_names, Args &...args) {
  if (used_node_names_.find(name) != used_node_names_.end()){
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
  return node;
}

template <typename T, typename... Args,
          typename std::enable_if<std::is_base_of<Node, T>{}, int>::type>
Node *Graph::createNode(const std::string &name,
                        std::vector<std::string> input_names,
                        std::vector<Edge *> outputs, Args &...args) {
  if (used_node_names_.find(name) != used_node_names_.end()){
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
  return node;
}

template <typename T, typename... Args,
          typename std::enable_if<std::is_base_of<Node, T>{}, int>::type>
Node *Graph::createNode(const std::string &name, std::vector<Edge *> inputs,
                        std::vector<std::string> output_names, Args &...args) {
  if (used_node_names_.find(name) != used_node_names_.end()){
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
  return node;
}

template <typename T, typename... Args,
          typename std::enable_if<std::is_base_of<Node, T>{}, int>::type>
Node *Graph::createNode(const std::string &name,
                        std::initializer_list<Edge *> inputs,
                        std::initializer_list<Edge *> outputs, Args &...args) {
  if (used_node_names_.find(name) != used_node_names_.end()){
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
  return node;
}

template <typename T, typename... Args,
          typename std::enable_if<std::is_base_of<Node, T>{}, int>::type>
Node *Graph::createNode(const std::string &name,
                        std::initializer_list<std::string> input_names,
                        std::initializer_list<std::string> output_names,
                        Args &...args) {
  if (used_node_names_.find(name) != used_node_names_.end()){
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
  return node;
}

template <typename T, typename... Args,
          typename std::enable_if<std::is_base_of<Node, T>{}, int>::type>
Node *Graph::createNode(const std::string &name,
                        std::initializer_list<Edge *> inputs,
                        std::initializer_list<std::string> output_names,
                        Args &...args) {
  if (used_node_names_.find(name) != used_node_names_.end()){
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
  return node;
}

template <typename T, typename... Args,
          typename std::enable_if<std::is_base_of<Node, T>{}, int>::type>
Node *Graph::createNode(const std::string &name,
                        std::initializer_list<std::string> input_names,
                        std::initializer_list<Edge *> outputs, Args &...args) {
  if (used_node_names_.find(name) != used_node_names_.end()){
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
  return node;
}

template <typename T, typename... Args,
          typename std::enable_if<std::is_base_of<Node, T>{}, int>::type>
Node *Graph::createInfer(const std::string &name, base::InferenceType type,
                         Edge *input, Edge *output) {
  Node *node = dynamic_cast<Node *>(new T(name, type, input, output));
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
  return node;
}

template <typename T, typename... Args,
          typename std::enable_if<std::is_base_of<Node, T>{}, int>::type>
Node *Graph::createInfer(const std::string &name, base::InferenceType type,
                         const std::string &input_name,
                         const std::string &output_name) {
  Edge *input = getEdge(input_name);
  if (input == nullptr) {
    input = createEdge(input_name);
  }
  Edge *output = getEdge(output_name);
  if (output == nullptr) {
    output = createEdge(output_name);
  }
  Node *node = dynamic_cast<Node *>(new T(name, type, input, output));
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
  return node;
}

template <typename T, typename... Args,
          typename std::enable_if<std::is_base_of<Node, T>{}, int>::type>
Node *Graph::createInfer(const std::string &name, base::InferenceType type,
                         Edge *input, const std::string &output_name) {
  Edge *output = getEdge(output_name);
  if (output == nullptr) {
    output = createEdge(output_name);
  }
  Node *node = dynamic_cast<Node *>(new T(name, type, input, output));
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
  return node;
}

template <typename T, typename... Args,
          typename std::enable_if<std::is_base_of<Node, T>{}, int>::type>
Node *Graph::createInfer(const std::string &name, base::InferenceType type,
                         const std::string &input_name, Edge *output) {
  Edge *input = getEdge(input_name);
  if (input == nullptr) {
    input = createEdge(input_name);
  }
  Node *node = dynamic_cast<Node *>(new T(name, type, input, output));
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
  return node;
}

template <typename T, typename... Args,
          typename std::enable_if<std::is_base_of<Node, T>{}, int>::type>
Node *Graph::createInfer(const std::string &name, base::InferenceType type,
                         std::vector<Edge *> inputs,
                         std::vector<Edge *> outputs) {
  Node *node = dynamic_cast<Node *>(new T(name, type, inputs, outputs));
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
  return node;
}

template <typename T, typename... Args,
          typename std::enable_if<std::is_base_of<Node, T>{}, int>::type>
Node *Graph::createInfer(const std::string &name, base::InferenceType type,
                         std::vector<std::string> input_names,
                         std::vector<std::string> output_names) {
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
  Node *node = dynamic_cast<Node *>(new T(name, type, inputs, outputs));
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
  return node;
}

template <typename T, typename... Args,
          typename std::enable_if<std::is_base_of<Node, T>{}, int>::type>
Node *Graph::createInfer(const std::string &name, base::InferenceType type,
                         std::vector<Edge *> inputs,
                         std::vector<std::string> output_names) {
  std::vector<Edge *> outputs;
  for (auto output_name : output_names) {
    Edge *output = getEdge(output_name);
    if (output == nullptr) {
      output = createEdge(output_name);
    }
    outputs.emplace_back(output);
  }
  Node *node = dynamic_cast<Node *>(new T(name, type, inputs, outputs));
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
  return node;
}

template <typename T, typename... Args,
          typename std::enable_if<std::is_base_of<Node, T>{}, int>::type>
Node *Graph::createInfer(const std::string &name, base::InferenceType type,
                         std::vector<std::string> input_names,
                         std::vector<Edge *> outputs) {
  std::vector<Edge *> inputs;
  for (auto input_name : input_names) {
    Edge *input = getEdge(input_name);
    if (input == nullptr) {
      input = createEdge(input_name);
    }
    inputs.emplace_back(input);
  }
  Node *node = dynamic_cast<Node *>(new T(name, type, inputs, outputs));
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
  return node;
}

template <typename T, typename... Args,
          typename std::enable_if<std::is_base_of<Node, T>{}, int>::type>
Node *Graph::createInfer(const std::string &name, base::InferenceType type,
                         std::initializer_list<Edge *> inputs,
                         std::initializer_list<Edge *> outputs) {
  Node *node = dynamic_cast<Node *>(new T(name, type, inputs, outputs));
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
  return node;
}

template <typename T, typename... Args,
          typename std::enable_if<std::is_base_of<Node, T>{}, int>::type>
Node *Graph::createInfer(const std::string &name, base::InferenceType type,
                         std::initializer_list<std::string> input_names,
                         std::initializer_list<std::string> output_names) {
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
  Node *node = dynamic_cast<Node *>(new T(name, type, inputs, outputs));
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
  return node;
}

template <typename T, typename... Args,
          typename std::enable_if<std::is_base_of<Node, T>{}, int>::type>
Node *Graph::createInfer(const std::string &name, base::InferenceType type,
                         std::initializer_list<Edge *> inputs,
                         std::initializer_list<std::string> output_names) {
  std::vector<Edge *> outputs;
  for (auto output_name : output_names) {
    Edge *output = getEdge(output_name);
    if (output == nullptr) {
      output = createEdge(output_name);
    }
    outputs.emplace_back(output);
  }
  Node *node = dynamic_cast<Node *>(new T(name, type, inputs, outputs));
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
  return node;
}

template <typename T, typename... Args,
          typename std::enable_if<std::is_base_of<Node, T>{}, int>::type>
Node *Graph::createInfer(const std::string &name, base::InferenceType type,
                         std::initializer_list<std::string> input_names,
                         std::initializer_list<Edge *> outputs) {
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
  Node *node = dynamic_cast<Node *>(new T(name, type, inputs, outputs_vec));
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
