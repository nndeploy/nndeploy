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

  base::Status setImageUrl(const std::string &key, const std::string &url);
  base::Status removeImageUrl(const std::string &key);
  base::Status setVideoUrl(const std::string &key, const std::string &url);
  base::Status removeVideoUrl(const std::string &key);
  base::Status setAudioUrl(const std::string &key, const std::string &url);
  base::Status removeAudioUrl(const std::string &key);
  base::Status setModelUrl(const std::string &key, const std::string &url);
  base::Status removeModelUrl(const std::string &key);
  base::Status setOtherUrl(const std::string &key, const std::string &url);
  base::Status removeOtherUrl(const std::string &key);
  std::string getImageUrl(const std::string &key);
  std::string getVideoUrl(const std::string &key);
  std::string getAudioUrl(const std::string &key);
  std::string getModelUrl(const std::string &key);
  std::string getOtherUrl(const std::string &key);

  base::Status setEdgeQueueMaxSize(int queue_max_size);
  int getEdgeQueueMaxSize();

  virtual base::Status setInput(Edge *input, int index = -1);
  virtual base::Status setOutput(Edge *output, int index = -1);

  virtual base::Status setInputs(std::vector<Edge *> inputs);
  virtual base::Status setOutputs(std::vector<Edge *> outputs);

  virtual base::Status setInputSharedPtr(std::shared_ptr<Edge> input,
                                         int index = -1);
  virtual base::Status setOutputSharedPtr(std::shared_ptr<Edge> output,
                                          int index = -1);

  virtual base::Status setInputsSharedPtr(
      std::vector<std::shared_ptr<Edge>> inputs);
  virtual base::Status setOutputsSharedPtr(
      std::vector<std::shared_ptr<Edge>> outputs);

  // create edge
  Edge *createEdge(const std::string &name);
  std::shared_ptr<Edge> createEdgeSharedPtr(const std::string &name);

  // add edge
  EdgeWrapper *addEdge(Edge *edge, bool is_external = true);
  EdgeWrapper *addEdgeSharedPtr(std::shared_ptr<Edge> edge);

  // get edge
  Edge *getEdge(const std::string &name);
  std::shared_ptr<Edge> getEdgeSharedPtr(const std::string &name);

  // update edge
  base::Status updteEdge(EdgeWrapper *edge_wrapper, Edge *edge,
                         bool is_external = true);

  // create node
  Node *createNode(const std::string &key, const std::string &name = "");
  Node *createNode(const NodeDesc &desc);

  template <typename T, typename... Args,
            typename std::enable_if<std::is_base_of<Node, T>{}, int>::type = 0>
  Node *createNode(const std::string &name = "", Args &...args);
  template <typename T, typename... Args,
            typename std::enable_if<std::is_base_of<Node, T>{}, int>::type = 0>
  Node *createNode(const NodeDesc &desc, Args &...args);

  base::Status setNodeDesc(Node *node, const NodeDesc &desc);

  // add node
  base::Status addNode(Node *node, bool is_external = true);
  base::Status addNodeSharedPtr(std::shared_ptr<Node> node);

  // get node
  Node *getNode(const std::string &name);
  Node *getNode(int index);
  std::shared_ptr<Node> getNodeSharedPtr(const std::string &name);
  Node *getNodeByKey(const std::string &key);
  std::vector<Node *> getNodesByKey(const std::string &key);
  int getNodeCount();
  std::vector<Node *> getNodes();
  std::vector<Node *> getNodesRecursive();
  std::vector<std::string> getNodesName();
  std::vector<std::string> getNodesNameRecursive();

  std::map<std::string, std::shared_ptr<RunStatus>> getNodesRunStatus();
  std::map<std::string, std::shared_ptr<RunStatus>>
  getNodesRunStatusRecursive();

  // help function
  base::Status addNodeInputAndOutput(NodeWrapper *node_wrapper,
                                     std::vector<Edge *> inputs,
                                     std::vector<Edge *> outputs);

  // set node param
  base::Status setNodeParam(const std::string &node_name, base::Param *param);
  base::Param *getNodeParam(const std::string &node_name);
  base::Status setNodeParamSharedPtr(const std::string &node_name,
                                     std::shared_ptr<base::Param> param);
  std::shared_ptr<base::Param> getNodeParamSharedPtr(
      const std::string &node_name);

  base::Status setExternalParam(const std::string &key,
                                std::shared_ptr<base::Param> param);
  std::shared_ptr<base::Param> getExternalParam(const std::string &key);

  base::Status setNodeParallelType(const std::string &node_name,
                                   base::ParallelType parallel_type);

  // set graph node share stream
  void setGraphNodeShareStream(bool flag);
  bool getGraphNodeShareStream();

  // set graph loop count
  virtual void setLoopMaxFlag(bool is_loop_max_flag);
  virtual bool getLoopMaxFlag();
  virtual void setLoopCount(int loop_count);
  virtual int getLoopCount();
  virtual std::map<std::string, int> getLoopCountMap();

  // update node io
  base::Status updateNodeIO(Node *node, std::vector<Edge *> inputs,
                            std::vector<Edge *> outputs);
  base::Status markInputEdge(std::vector<Edge *> inputs);
  base::Status markOutputEdge(std::vector<Edge *> outputs);

  virtual base::Status defaultParam();

  virtual base::Status init();
  virtual base::Status deinit();

  virtual base::Status run();
  virtual bool synchronize();

  // This method must be implemented by subclasses
  // Subclasses should override this method to define their own operator()
  // implementation
  virtual std::vector<Edge *> forward(std::vector<Edge *> inputs);
  virtual std::vector<Edge *> operator()(std::vector<Edge *> inputs);
  virtual std::vector<Edge *> forward();
  virtual std::vector<Edge *> operator()();
  virtual std::vector<Edge *> forward(Edge *input);
  virtual std::vector<Edge *> operator()(Edge *input);

  base::Status dump(std::ostream &oss = std::cout);

  virtual void setTraceFlag(bool flag);
  std::vector<Edge *> trace(std::vector<Edge *> inputs);
  std::vector<Edge *> trace();
  std::vector<Edge *> trace(Edge *input);

  bool isForwardApiOk();
  base::Status toStaticGraph();

  // create node
  // Not recommended api
  template <typename T, typename... Args,
            typename std::enable_if<std::is_base_of<Node, T>{}, int>::type = 0>
  Node *createNode(const std::string &name, Edge *input, Edge *output,
                   Args &...args);
  template <typename T, typename... Args,
            typename std::enable_if<std::is_base_of<Node, T>{}, int>::type = 0>
  Node *createNode(const std::string &name, const std::string &input_name,
                   const std::string &output_name, Args &...args);
  template <typename T, typename... Args,
            typename std::enable_if<std::is_base_of<Node, T>{}, int>::type = 0>
  Node *createNode(const std::string &name, Edge *input,
                   const std::string &output_name, Args &...args);
  template <typename T, typename... Args,
            typename std::enable_if<std::is_base_of<Node, T>{}, int>::type = 0>
  Node *createNode(const std::string &name, const std::string &input_name,
                   Edge *output, Args &...args);
  template <typename T, typename... Args,
            typename std::enable_if<std::is_base_of<Node, T>{}, int>::type = 0>
  Node *createNode(const std::string &name, std::vector<Edge *> inputs,
                   std::vector<Edge *> outputs, Args &...args);
  template <typename T, typename... Args,
            typename std::enable_if<std::is_base_of<Node, T>{}, int>::type = 0>
  Node *createNode(const std::string &name,
                   std::vector<std::string> input_names,
                   std::vector<std::string> output_names, Args &...args);
  template <typename T, typename... Args,
            typename std::enable_if<std::is_base_of<Node, T>{}, int>::type = 0>
  Node *createNode(const std::string &name,
                   std::vector<std::string> input_names,
                   std::vector<Edge *> outputs, Args &...args);
  template <typename T, typename... Args,
            typename std::enable_if<std::is_base_of<Node, T>{}, int>::type = 0>
  Node *createNode(const std::string &name, std::vector<Edge *> inputs,
                   std::vector<std::string> output_names, Args &...args);
  template <typename T, typename... Args,
            typename std::enable_if<std::is_base_of<Node, T>{}, int>::type = 0>
  Node *createNode(const std::string &name,
                   std::initializer_list<Edge *> inputs,
                   std::initializer_list<Edge *> outputs, Args &...args);
  template <typename T, typename... Args,
            typename std::enable_if<std::is_base_of<Node, T>{}, int>::type = 0>
  Node *createNode(const std::string &name,
                   std::initializer_list<std::string> input_names,
                   std::initializer_list<std::string> output_names,
                   Args &...args);
  template <typename T, typename... Args,
            typename std::enable_if<std::is_base_of<Node, T>{}, int>::type = 0>
  Node *createNode(const std::string &name,
                   std::initializer_list<Edge *> inputs,
                   std::initializer_list<std::string> output_names,
                   Args &...args);
  template <typename T, typename... Args,
            typename std::enable_if<std::is_base_of<Node, T>{}, int>::type = 0>
  Node *createNode(const std::string &name,
                   std::initializer_list<std::string> input_names,
                   std::initializer_list<Edge *> outputs, Args &...args);
  template <typename T, typename... Args,
            typename std::enable_if<std::is_base_of<Node, T>{}, int>::type = 0>
  Node *createInfer(const std::string &name, base::InferenceType type,
                    Edge *input, Edge *output);
  template <typename T, typename... Args,
            typename std::enable_if<std::is_base_of<Node, T>{}, int>::type = 0>
  Node *createInfer(const std::string &name, base::InferenceType type,
                    const std::string &input_name,
                    const std::string &output_name);
  template <typename T, typename... Args,
            typename std::enable_if<std::is_base_of<Node, T>{}, int>::type = 0>
  Node *createInfer(const std::string &name, base::InferenceType type,
                    Edge *input, const std::string &output_name);
  template <typename T, typename... Args,
            typename std::enable_if<std::is_base_of<Node, T>{}, int>::type = 0>
  Node *createInfer(const std::string &name, base::InferenceType type,
                    const std::string &input_name, Edge *output);
  template <typename T, typename... Args,
            typename std::enable_if<std::is_base_of<Node, T>{}, int>::type = 0>
  Node *createInfer(const std::string &name, base::InferenceType type,
                    std::vector<Edge *> inputs, std::vector<Edge *> outputs);
  template <typename T, typename... Args,
            typename std::enable_if<std::is_base_of<Node, T>{}, int>::type = 0>
  Node *createInfer(const std::string &name, base::InferenceType type,
                    std::vector<std::string> input_names,
                    std::vector<std::string> output_names);
  template <typename T, typename... Args,
            typename std::enable_if<std::is_base_of<Node, T>{}, int>::type = 0>
  Node *createInfer(const std::string &name, base::InferenceType type,
                    std::vector<Edge *> inputs,
                    std::vector<std::string> output_names);
  template <typename T, typename... Args,
            typename std::enable_if<std::is_base_of<Node, T>{}, int>::type = 0>
  Node *createInfer(const std::string &name, base::InferenceType type,
                    std::vector<std::string> input_names,
                    std::vector<Edge *> outputs);
  template <typename T, typename... Args,
            typename std::enable_if<std::is_base_of<Node, T>{}, int>::type = 0>
  Node *createInfer(const std::string &name, base::InferenceType type,
                    std::initializer_list<Edge *> inputs,
                    std::initializer_list<Edge *> outputs);
  template <typename T, typename... Args,
            typename std::enable_if<std::is_base_of<Node, T>{}, int>::type = 0>
  Node *createInfer(const std::string &name, base::InferenceType type,
                    std::initializer_list<std::string> input_names,
                    std::initializer_list<std::string> output_names);
  template <typename T, typename... Args,
            typename std::enable_if<std::is_base_of<Node, T>{}, int>::type = 0>
  Node *createInfer(const std::string &name, base::InferenceType type,
                    std::initializer_list<Edge *> inputs,
                    std::initializer_list<std::string> output_names);
  template <typename T, typename... Args,
            typename std::enable_if<std::is_base_of<Node, T>{}, int>::type = 0>
  Node *createInfer(const std::string &name, base::InferenceType type,
                    std::initializer_list<std::string> input_names,
                    std::initializer_list<Edge *> outputs);
  template <typename T, typename... Args,
            typename std::enable_if<std::is_base_of<Node, T>{}, int>::type = 0>
  Node *createInfer(const NodeDesc &desc, base::InferenceType type);
  Node *createNode4Py(const std::string &key, const std::string &name = "");
  Node *createNode4Py(const NodeDesc &desc);

  EdgeWrapper *getEdgeWrapper(Edge *edge);
  EdgeWrapper *getEdgeWrapper(const std::string &name);

  NodeWrapper *getNodeWrapper(Node *node);
  NodeWrapper *getNodeWrapper(const std::string &name);

  // to json
  // using Node::serialize;
  virtual base::Status serialize(rapidjson::Value &json,
                                 rapidjson::Document::AllocatorType &allocator);
  virtual std::string serialize();
  // from json
  // using Node::deserialize;
  virtual base::Status deserialize(rapidjson::Value &json);
  virtual base::Status deserialize(const std::string &json_str);

 protected:
  virtual base::Status construct();
  virtual base::Status executor();

 protected:
  // 
  std::map<std::string, std::string> image_url_;
  std::map<std::string, std::string> video_url_;
  std::map<std::string, std::string> audio_url_;
  std::map<std::string, std::string> model_url_;
  std::map<std::string, std::string> other_url_;

  bool is_graph_node_share_stream_ = true;
  std::vector<EdgeWrapper *> edge_repository_;
  std::vector<NodeWrapper *> node_repository_;
  std::vector<NodeWrapper *> run_node_repository_;
  std::vector<std::shared_ptr<Edge>> shared_edge_repository_;
  std::vector<std::shared_ptr<Node>> shared_node_repository_;
  std::set<std::string> used_node_names_;
  std::set<std::string> used_edge_names_;
  std::shared_ptr<Executor> executor_;
  int queue_max_size_ = 16;
  std::map<std::string, std::shared_ptr<base::Param>>
      external_param_repository_;
  bool is_loop_max_flag_ = true;
  bool is_forward_api_ok_ = true;
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

template <typename T, typename... Args,
          typename std::enable_if<std::is_base_of<Node, T>{}, int>::type>
Node *Graph::createInfer(const NodeDesc &desc, base::InferenceType type) {
  return this->createInfer<T>(desc.getName(), type, desc.getInputs(),
                              desc.getOutputs());
}

// template <typename... Args>
// Node *Graph::createNode(const NodeDesc &desc, Args &...args) {
//   const std::string &name = desc.getName();
//   const std::string &node_key = desc.getKey();
//   std::vector<std::string> input_names = desc.getInputs();
//   std::vector<std::string> output_names = desc.getOutputs();
//   if (used_node_names_.find(name) != used_node_names_.end()) {
//     NNDEPLOY_LOGE("node name[%s] is already used!\n", name.c_str());
//     return nullptr;
//   }
//   std::vector<Edge *> inputs;
//   for (auto input_name : input_names) {
//     Edge *input = getEdge(input_name);
//     if (input == nullptr) {
//       input = createEdge(input_name);
//     }
//     inputs.emplace_back(input);
//   }
//   std::vector<Edge *> outputs;
//   for (auto output_name : output_names) {
//     Edge *output = getEdge(output_name);
//     if (output == nullptr) {
//       output = createEdge(output_name);
//     }
//     outputs.emplace_back(output);
//   }
//   Node *node = nndeploy::dag::createNode(node_key, name, inputs, outputs);
//   // Node *node =
//   //     nndeploy::dag::createNode(node_key, name, inputs, outputs, args...);
//   if (node == nullptr) {
//     NNDEPLOY_LOGE("create infer node[%s] failed!\n", desc.getName().c_str());
//     return nullptr;
//   }
//   NodeWrapper *node_wrapper = new NodeWrapper();
//   node_wrapper->is_external_ = false;
//   node_wrapper->node_ = node;
//   node_wrapper->name_ = name;
//   for (auto input : inputs) {
//     EdgeWrapper *input_wrapper = findEdgeWrapper(edge_repository_, input);
//     if (input_wrapper == nullptr) {
//       input_wrapper = this->addEdge(input);
//     }
//     input_wrapper->consumers_.emplace_back(node_wrapper);
//   }
//   for (auto output : outputs) {
//     EdgeWrapper *output_wrapper = findEdgeWrapper(edge_repository_, output);
//     if (output_wrapper == nullptr) {
//       output_wrapper = this->addEdge(output);
//     }
//     output_wrapper->producers_.emplace_back(node_wrapper);
//   }

//   node_repository_.emplace_back(node_wrapper);
//   used_node_names_.insert(name);

//   node->setGraph(this);

//   return node;
// }

// template <typename T, typename... Args,
//           typename std::enable_if<std::is_base_of<Node, T>{}, int>::type = 0>
// Node *Graph::createNode(const std::string &name = "", Args &...args) {
//   Node *node = this->createNode<T>(name, {}, {}, args...);
//   if (node == nullptr) {
//     NNDEPLOY_LOGE("create node[%s] failed!\n", name.c_str());
//     return nullptr;
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
  return node;
}

// Not recommended api
using createGraphFunc = std::function<Graph *(
    const std::string &name, base::InferenceType inference_type,
    base::DeviceType device_type, Edge *input, Edge *output,
    base::ModelType model_type, bool is_path,
    std::vector<std::string> model_value)>;

extern NNDEPLOY_CC_API std::map<std::string, createGraphFunc> &
getGlobalGraphCreatorMap();

class NNDEPLOY_CC_API TypeGraphRegister{public : explicit TypeGraphRegister(
    const std::string &name,
    createGraphFunc func){getGlobalGraphCreatorMap()[name] = func;
}  // namespace dag
};  // namespace nndeploy

extern NNDEPLOY_CC_API Graph *createGraph(const std::string &name,
                                          base::InferenceType inference_type,
                                          base::DeviceType device_type,
                                          Edge *input, Edge *output,
                                          base::ModelType model_type,
                                          bool is_path,
                                          std::vector<std::string> model_value);

// to json
extern NNDEPLOY_CC_API base::Status serialize(
    Graph *graph, rapidjson::Value &json,
    rapidjson::Document::AllocatorType &allocator);
extern NNDEPLOY_CC_API std::string serialize(Graph *graph);
extern NNDEPLOY_CC_API base::Status saveFile(Graph *graph,
                                             const std::string &path);
// from json
extern NNDEPLOY_CC_API Graph *deserialize(rapidjson::Value &json);
extern NNDEPLOY_CC_API Graph *deserialize(const std::string &json_str);
extern NNDEPLOY_CC_API Graph *loadFile(const std::string &path);

}  // namespace dag
}  // namespace nndeploy

#endif  // _NNDEPLOY_DAG_EXECUTOR_H_
