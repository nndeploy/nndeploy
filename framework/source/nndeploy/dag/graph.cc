
#include "nndeploy/dag/graph.h"

#include "nndeploy/base/any.h"
#include "nndeploy/base/common.h"
#include "nndeploy/base/glic_stl_include.h"
#include "nndeploy/base/log.h"
#include "nndeploy/base/macro.h"
#include "nndeploy/base/object.h"
#include "nndeploy/base/status.h"
#include "nndeploy/base/string.h"
#include "nndeploy/base/time_profiler.h"
#include "nndeploy/dag/edge.h"
#include "nndeploy/dag/executor/parallel_pipeline_executor.h"
#include "nndeploy/dag/executor/parallel_task_executor.h"
#include "nndeploy/dag/executor/sequential_executor.h"
#include "nndeploy/dag/node.h"
#include "nndeploy/dag/util.h"
#include "nndeploy/device/buffer.h"
#include "nndeploy/device/device.h"
#include "nndeploy/device/memory_pool.h"
#include "nndeploy/device/tensor.h"

namespace nndeploy {
namespace dag {

Graph::Graph(const std::string &name) : Node(name) {
  key_ = "nndeploy::dag::Graph";
}
Graph::Graph(const std::string &name, std::vector<Edge *> inputs,
             std::vector<Edge *> outputs)
    : Node(name, inputs, outputs) {
  key_ = "nndeploy::dag::Graph";
  for (auto input : inputs) {
    if (nullptr == addEdge(input)) {
      constructed_ = false;
      return;
    }
  }
  for (auto output : outputs) {
    if (nullptr == addEdge(output)) {
      constructed_ = false;
      return;
    }
  }
  constructed_ = true;
  is_graph_ = true;
}
Graph::~Graph() {
  for (auto node_wrapper : node_repository_) {
    if (!node_wrapper->is_external_) {
      delete node_wrapper->node_;
    }
    delete node_wrapper;
  }
  for (auto edge_wrapper : edge_repository_) {
    if (!edge_wrapper->is_external_) {
      delete edge_wrapper->edge_;
    }
    delete edge_wrapper;
  }
  node_repository_.clear();
  used_node_names_.clear();
  edge_repository_.clear();
  used_edge_names_.clear();
  shared_edge_repository_.clear();
  shared_node_repository_.clear();
}

base::Status Graph::setEdgeQueueMaxSize(int queue_max_size) {
  queue_max_size_ = queue_max_size;
  return base::kStatusCodeOk;
}
int Graph::getEdgeQueueMaxSize() { return queue_max_size_; }

Edge *Graph::createEdge(const std::string &name) {
  std::string unique_name = name;
  if (unique_name.empty()) {
    unique_name = "edge_" + base::getUniqueString();
  }
  if (used_edge_names_.find(unique_name) != used_edge_names_.end()) {
    NNDEPLOY_LOGE("edge name[%s] is already used!\n", unique_name.c_str());
    return nullptr;
  }
  Edge *edge = new Edge(unique_name);
  EdgeWrapper *edge_wrapper = new EdgeWrapper();
  edge_wrapper->is_external_ = false;
  edge_wrapper->edge_ = edge;
  edge_wrapper->name_ = unique_name;
  edge_repository_.emplace_back(edge_wrapper);
  used_edge_names_.insert(unique_name);
  return edge;
}

std::shared_ptr<Edge> Graph::createEdgeSharedPtr(const std::string &name) {
  std::string unique_name = name;
  if (unique_name.empty()) {
    unique_name = "edge_" + base::getUniqueString();
  }
  if (used_edge_names_.find(unique_name) != used_edge_names_.end()) {
    NNDEPLOY_LOGE("edge name[%s] is already used!\n", unique_name.c_str());
    return nullptr;
  }
  std::shared_ptr<Edge> edge_ptr = std::make_shared<Edge>(unique_name);
  Edge *edge = edge_ptr.get();
  EdgeWrapper *edge_wrapper = new EdgeWrapper();
  // 创建shared edge
  edge_wrapper->is_external_ = true;
  edge_wrapper->edge_ = edge;
  edge_wrapper->name_ = unique_name;
  edge_repository_.emplace_back(edge_wrapper);
  used_edge_names_.insert(unique_name);

  shared_edge_repository_.emplace_back(edge_ptr);
  return edge_ptr;
}

Edge *Graph::getEdge(const std::string &name) {
  for (EdgeWrapper *edge_wrapper : edge_repository_) {
    if (edge_wrapper->name_ == name) {
      return edge_wrapper->edge_;
    }
  }
  return nullptr;
}

std::shared_ptr<Edge> Graph::getEdgeSharedPtr(const std::string &name) {
  for (auto edge_ptr : shared_edge_repository_) {
    if (edge_ptr->getName() == name) {
      return edge_ptr;
    }
  }
  return nullptr;
}

EdgeWrapper *Graph::addEdge(Edge *edge, bool is_external) {
  NNDEPLOY_CHECK_PARAM_NULL_RET_NULL(edge, "edge is null!");
  if (used_edge_names_.find(edge->getName()) != used_edge_names_.end()) {
    for (auto edge_wrapper : edge_repository_) {
      if (edge_wrapper->edge_ == edge) {
        return edge_wrapper;
      }
    }
  }
  EdgeWrapper *edge_wrapper = new EdgeWrapper();
  edge_wrapper->is_external_ = is_external;
  edge_wrapper->edge_ = edge;
  edge_wrapper->name_ = edge->getName();
  edge_repository_.emplace_back(edge_wrapper);
  used_edge_names_.insert(edge->getName());
  return edge_wrapper;
}

EdgeWrapper *Graph::addEdgeSharedPtr(std::shared_ptr<Edge> edge) {
  if (edge == nullptr) {
    NNDEPLOY_LOGE("edge is null!");
    return nullptr;
  }
  EdgeWrapper *edge_wrapper = this->addEdge(edge.get(), true);
  if (edge_wrapper == nullptr) {
    NNDEPLOY_LOGE("addEdge failed!");
    return nullptr;
  }
  shared_edge_repository_.emplace_back(edge);
  return edge_wrapper;
}

base::Status Graph::updteEdge(EdgeWrapper *edge_wrapper, Edge *edge,
                              bool is_external) {
  NNDEPLOY_CHECK_PARAM_NULL_RET_STATUS(edge, "edge is null!");
  // 从shared_edge_repository_中移除
  auto shared_it = std::find_if(
      shared_edge_repository_.begin(), shared_edge_repository_.end(),
      [edge_wrapper](std::shared_ptr<Edge> &shared_edge) {
        return shared_edge.get() == edge_wrapper->edge_;
      });
  if (shared_it != shared_edge_repository_.end()) {
    shared_edge_repository_.erase(shared_it);
  }
  if (!edge_wrapper->is_external_) {
    delete edge_wrapper->edge_;
  }
  edge_wrapper->edge_ = edge;
  edge_wrapper->is_external_ = is_external;
  return base::kStatusCodeOk;
}

Node *Graph::createNode(const NodeDesc &desc) {
  const std::string &name = desc.getName();
  const std::string &node_key = desc.getKey();
  std::vector<std::string> input_names = desc.getInputs();
  std::vector<std::string> output_names = desc.getOutputs();
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
  Node *node = nndeploy::dag::createNode(node_key, name, inputs, outputs);
  if (node == nullptr) {
    NNDEPLOY_LOGE("create node[%s] failed!\n", desc.getName().c_str());
    return nullptr;
  }
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

base::Status Graph::addNode(Node *node, bool is_external) {
  NNDEPLOY_CHECK_PARAM_NULL_RET_STATUS(node, "node is null!");
  if (this == node) {
    NNDEPLOY_LOGE("Graph[%s] cannot add itself as node\n",
                  this->getName().c_str());
    return base::kStatusCodeErrorInvalidValue;
  }
  if (used_node_names_.find(node->getName()) != used_node_names_.end()) {
    NNDEPLOY_LOGW("Warning: node name[%s] is already used!\n",
                  node->getName().c_str());
  }
  base::Status status = base::kStatusCodeOk;
  NodeWrapper *node_wrapper = new NodeWrapper();
  node_wrapper->is_external_ = is_external;
  node_wrapper->node_ = node;
  node_wrapper->name_ = node->getName();
  for (auto input : node->getAllInput()) {
    EdgeWrapper *input_wrapper = findEdgeWrapper(edge_repository_, input);
    if (input_wrapper == nullptr) {
      input_wrapper = this->addEdge(input, is_external);
    }
    // input_wrapper->consumers_.emplace_back(node_wrapper);
    insertUnique(input_wrapper->consumers_, node_wrapper);
  }
  for (auto output : node->getAllOutput()) {
    EdgeWrapper *output_wrapper = findEdgeWrapper(edge_repository_, output);
    if (output_wrapper == nullptr) {
      output_wrapper = this->addEdge(output, is_external);
    }
    // output_wrapper->producers_.emplace_back(node_wrapper);
    insertUnique(output_wrapper->producers_, node_wrapper);
  }

  node_repository_.emplace_back(node_wrapper);
  used_node_names_.insert(node->getName());

  node->setGraph(this);

  return status;
}
base::Status Graph::addNodeSharedPtr(std::shared_ptr<Node> node) {
  if (node == nullptr) {
    NNDEPLOY_LOGE("node is null!");
    return base::kStatusCodeErrorInvalidValue;
  }
  base::Status status = addNode(node.get(), true);
  NNDEPLOY_LOGE("addNodeSharedPtr: %s\n", node->getName().c_str());
  NNDEPLOY_RETURN_ON_NEQ(status, base::kStatusCodeOk, "addNode failed!");
  shared_node_repository_.emplace_back(node);
  return status;
}

Node *Graph::getNode(const std::string &name) {
  for (auto node_wrapper : node_repository_) {
    if (node_wrapper->name_ == name) {
      return node_wrapper->node_;
    }
  }
  return nullptr;
}

std::shared_ptr<Node> Graph::getNodeSharedPtr(const std::string &name) {
  for (auto node_ptr : shared_node_repository_) {
    if (node_ptr->getName() == name) {
      return node_ptr;
    }
  }
  return nullptr;
}

Node *Graph::getNodeByKey(const std::string &key) {
  for (auto node_wrapper : node_repository_) {
    if (node_wrapper->node_->getKey() == key) {
      return node_wrapper->node_;
    }
  }
  return nullptr;
}

std::vector<Node *> Graph::getNodesByKey(const std::string &key) {
  std::vector<Node *> nodes;
  for (auto node_wrapper : node_repository_) {
    if (node_wrapper->node_->getKey() == key) {
      nodes.emplace_back(node_wrapper->node_);
    }
  }
  return nodes;
}

base::Status Graph::setNodeParam(const std::string &node_name,
                                 base::Param *param) {
  base::Status status = base::kStatusCodeOk;
  NNDEPLOY_CHECK_PARAM_NULL_RET_STATUS(param, "param is null!");
  NodeWrapper *node_wrapper = findNodeWrapper(node_repository_, node_name);
  NNDEPLOY_CHECK_PARAM_NULL_RET_STATUS(node_wrapper, "node_wrapper is null!");
  status = node_wrapper->node_->setParam(param);
  return status;
}

base::Param *Graph::getNodeParam(const std::string &node_name) {
  NodeWrapper *node_wrapper = findNodeWrapper(node_repository_, node_name);
  NNDEPLOY_CHECK_PARAM_NULL_RET_NULL(node_wrapper, "node_wrapper is null!");
  return node_wrapper->node_->getParam();
}

base::Status Graph::setNodeParamSharedPtr(const std::string &node_name,
                                          std::shared_ptr<base::Param> param) {
  NodeWrapper *node_wrapper = findNodeWrapper(node_repository_, node_name);
  NNDEPLOY_CHECK_PARAM_NULL_RET_STATUS(node_wrapper, "node_wrapper is null!");
  base::Status status = node_wrapper->node_->setParamSharedPtr(param);
  return status;
}
std::shared_ptr<base::Param> Graph::getNodeParamSharedPtr(
    const std::string &node_name) {
  NodeWrapper *node_wrapper = findNodeWrapper(node_repository_, node_name);
  NNDEPLOY_CHECK_PARAM_NULL_RET_NULL(node_wrapper, "node_wrapper is null!");
  return node_wrapper->node_->getParamSharedPtr();
}

base::Status Graph::setNodeParallelType(const std::string &node_name,
                                        base::ParallelType parallel_type) {
  NodeWrapper *node_wrapper = findNodeWrapper(node_repository_, node_name);
  NNDEPLOY_CHECK_PARAM_NULL_RET_STATUS(node_wrapper, "node_wrapper is null!");
  base::Status status = node_wrapper->node_->setParallelType(parallel_type);
  return status;
}

void Graph::setGraphNodeShareStream(bool flag) {
  is_graph_node_share_stream_ = flag;
}

bool Graph::getGraphNodeShareStream() { return is_graph_node_share_stream_; }

base::Status Graph::updateNodeIO(Node *node, std::vector<Edge *> inputs,
                                 std::vector<Edge *> outputs) {
  base::Status status = base::kStatusCodeOk;
  // 找到node对应的node_wrapper
  NodeWrapper *node_wrapper = nullptr;
  for (auto wrapper : node_repository_) {
    if (wrapper->node_ == node) {
      node_wrapper = wrapper;
      break;
    }
  }
  if (node_wrapper == nullptr) {
    NNDEPLOY_LOGE("can't find node_wrapper!");
    return base::kStatusCodeErrorInvalidValue;
  }
  for (auto input : inputs) {
    EdgeWrapper *edge_wrapper =
        findEdgeWrapper(edge_repository_, input->getName());
    if (edge_wrapper == nullptr) {
      edge_wrapper = this->addEdge(input, true);
      if (edge_wrapper == nullptr) {
        NNDEPLOY_LOGE("addEdge failed!");
        return base::kStatusCodeErrorInvalidValue;
      }
    } else {
      if (edge_wrapper->edge_ != input) {
        NNDEPLOY_LOGI("updateEdge: %s\n", input->getName().c_str());
        updteEdge(edge_wrapper, input, true);
      }
    }
    // 添加消费者
    insertUnique(edge_wrapper->consumers_, node_wrapper);
    // 打印edge及其消费者信息
    // NNDEPLOY_LOGI("Edge: %s\n", input->getName().c_str());
    // NNDEPLOY_LOGI("Consumer: %s\n", node_wrapper->node_->getName().c_str());
  }
  for (auto output : outputs) {
    EdgeWrapper *edge_wrapper =
        findEdgeWrapper(edge_repository_, output->getName());
    if (edge_wrapper == nullptr) {
      edge_wrapper = this->addEdge(output, true);
      if (edge_wrapper == nullptr) {
        NNDEPLOY_LOGE("addEdge failed!");
        return base::kStatusCodeErrorInvalidValue;
      }
    } else {
      if (edge_wrapper->edge_ != output) {
        NNDEPLOY_LOGI("updateEdge: %s\n", output->getName().c_str());
        updteEdge(edge_wrapper, output, true);
      }
    }
    // 添加生产者
    insertUnique(edge_wrapper->producers_, node_wrapper);
    // 打印edge及其生产者信息
    // NNDEPLOY_LOGI("Edge: %s\n", output->getName().c_str());
    // NNDEPLOY_LOGI("Producer: %s\n", node_wrapper->node_->getName().c_str());
  }
  return status;
}

base::Status Graph::markInputEdge(std::vector<Edge *> inputs) {
  for (auto input : inputs) {
    insertUnique(inputs_, input);
  }
  return base::kStatusCodeOk;
};
base::Status Graph::markOutputEdge(std::vector<Edge *> outputs) {
  for (auto output : outputs) {
    insertUnique(outputs_, output);
  }
  return base::kStatusCodeOk;
};

base::Status Graph::init() {
  base::Status status = base::kStatusCodeOk;

  // NNDEPLOY_LOGI("###########################\n");
  NNDEPLOY_LOGI("setInitializedFlag false!\n");
  // NNDEPLOY_LOGI("###########################\n");
  // setInitializedFlag(false);

  // NNDEPLOY_LOGE("###########################\n");
  // NNDEPLOY_LOGE("construct!\n");
  // NNDEPLOY_LOGE("###########################\n");
  status = this->construct();
  NNDEPLOY_RETURN_ON_NEQ(status, base::kStatusCodeOk,
                         "graph construct failed!");

  // NNDEPLOY_LOGE("###########################\n");
  // NNDEPLOY_LOGE("executor!\n");
  // NNDEPLOY_LOGE("###########################\n");
  status = this->executor();
  NNDEPLOY_RETURN_ON_NEQ(status, base::kStatusCodeOk, "graph executor failed!");

  // NNDEPLOY_LOGI("###########################\n");
  NNDEPLOY_LOGI("setInitializedFlag true!\n");
  // NNDEPLOY_LOGI("###########################\n");
  setInitializedFlag(true);

  return status;
}

base::Status Graph::deinit() {
  base::Status status = base::kStatusCodeOk;

  // NNDEPLOY_LOGI("#######################\n");
  // NNDEPLOY_LOGI("Node DeInitialize Phase!\n");
  // NNDEPLOY_LOGI("#######################\n");
  if (executor_ != nullptr) {
    status = executor_->deinit();
    NNDEPLOY_RETURN_ON_NEQ(status, base::kStatusCodeOk,
                           "executor deinit failed!");
  } else {
    for (auto node_wrapper : node_repository_) {
      if (node_wrapper->node_->getInitialized()) {
        node_wrapper->node_->deinit();
        node_wrapper->node_->setInitializedFlag(false);
      }
    }
  }

  // NNDEPLOY_LOGI("###########################\n");
  // NNDEPLOY_LOGI("setInitializedFlag false!\n");
  // NNDEPLOY_LOGI("###########################\n");
  setInitializedFlag(false);

  return status;
}

base::Status Graph::run() {
  base::Status status = base::kStatusCodeOk;

  // NNDEPLOY_LOGI("###########################\n");
  // NNDEPLOY_LOGI("setRunningFlag true!\n");
  // NNDEPLOY_LOGI("###########################\n");
  setRunningFlag(true);

  // NNDEPLOY_LOGI("#######################\n");
  // NNDEPLOY_LOGI("Node run Phase!\n");
  // NNDEPLOY_LOGI("#######################\n");
  status = executor_->run();
  NNDEPLOY_RETURN_ON_NEQ(status, base::kStatusCodeOk, "executor run failed!");

  // NNDEPLOY_LOGI("###########################\n");
  // NNDEPLOY_LOGI("setRunningFlag false!\n");
  // NNDEPLOY_LOGI("###########################\n");
  setRunningFlag(false);

  return status;
}

std::vector<Edge *> Graph::forward(std::vector<Edge *> inputs) {
  std::vector<Edge *> outputs;
  return outputs;
};
std::vector<Edge *> Graph::operator()(std::vector<Edge *> inputs) {
  if (traced_) {
    // NNDEPLOY_LOGI("graph traced!\n");
    base::Status status = this->run();
    if (status != base::kStatusCodeOk) {
      NNDEPLOY_LOGE("graph run failed!");
      return std::vector<Edge *>();
    }
    return outputs_;
  } else {
    // NNDEPLOY_LOGI("graph not traced!\n");
    this->markInputEdge(inputs);
    std::vector<Edge *> outputs = this->forward(inputs);
    if (graph_ != nullptr) {
      base::Status status = graph_->updateNodeIO(this, inputs, outputs);
      if (status != base::kStatusCodeOk) {
        NNDEPLOY_LOGE("graph_->updateNodeIO failed.\n");
        return std::vector<Edge *>();
      }
      // for (auto input : inputs) {
      //   NNDEPLOY_LOGE("input->getName(): %s.\n", input->getName().c_str());
      // }
      // for (auto output : outputs) {
      //   NNDEPLOY_LOGE("output->getName(): %s.\n", output->getName().c_str());
      // }
    }
    this->markOutputEdge(outputs);
    return outputs;
  }
}

base::Status Graph::dump(std::ostream &oss) {
  base::Status status = base::kStatusCodeOk;
  // start
  if (is_inner_) {
    if (name_.empty()) {
      oss << "subgraph cluster " << base::getUniqueString() << "{\n";
    } else {
      std::string label = "\"cluster " + name_ + "\"";
      oss << "subgraph " << label << " {\n";
      oss << "label = \"" + name_ + "\";\n";
    }
    oss << "color = blue;\n";
  } else {
    if (name_.empty()) {
      oss << "digraph graph {\n";
    } else {
      oss << "digraph " << name_ << " {\n";
    }
    for (auto input : inputs_) {
      if (input->getName().empty()) {
        oss << "p" << (void *)input << "[shape=diamond, label=input]\n";
      } else {
        std::string label = "\"" + input->getName() + "\"";
        oss << "p" << (void *)input << "[shape=diamond, label=" << label
            << "]\n";
      }
      EdgeWrapper *edge_wrapper = findEdgeWrapper(edge_repository_, input);
      std::vector<Node *> consumers;
      // for (auto consumer : edge_wrapper->consumers_) {
      //   auto consumer_node = consumer->node_;
      //   if (consumer_node->getGraphFlag()) {
      //     Graph *graph = (Graph *)consumer_node;
      //     EdgeWrapper *inner_edge_wrapper =
      //         graph->getEdgeWrapper(edge_wrapper->edge_);
      //     if (inner_edge_wrapper == nullptr) {
      //       // NNDEPLOY_LOGE("edge_wrapper[%s] is null!\n",
      //       //               edge_wrapper->name_.c_str());
      //       continue;
      //     }
      //     for (auto consumer : inner_edge_wrapper->consumers_) {
      //       consumers.emplace_back(consumer->node_);
      //     }
      //   } else {
      //     consumers.emplace_back(consumer_node);
      //   }
      // }
      findConsumerNode(edge_wrapper, consumers);
      for (auto node : consumers) {
        oss << "p" << (void *)input << "->"
            << "p" << (void *)node;
        if (input->getName().empty()) {
          oss << "\n";
        } else {
          std::string label = "\"" + input->getName() + "\"";
          oss << "[label=" << label << "]\n";
        }
      }
    }
    for (auto output : outputs_) {
      if (output->getName().empty()) {
        oss << "p" << (void *)output << "[shape=diamond, label=output]\n";
      } else {
        std::string label = "\"" + output->getName() + "\"";
        oss << "p" << (void *)output << "[shape=diamond, label=" << label
            << "]\n";
      }
      EdgeWrapper *edge_wrapper = findEdgeWrapper(edge_repository_, output);
      std::vector<Node *> producers;
      // for (auto producer : edge_wrapper->producers_) {
      //   auto producer_node = producer->node_;
      //   if (producer_node->getGraphFlag()) {
      //     Graph *graph = (Graph *)producer_node;
      //     EdgeWrapper *inner_edge_wrapper =
      //         graph->getEdgeWrapper(edge_wrapper->edge_);
      //     if (inner_edge_wrapper == nullptr) {
      //       // NNDEPLOY_LOGE("edge_wrapper[%s] is null!\n",
      //       //               edge_wrapper->name_.c_str());
      //       continue;
      //     }
      //     for (auto producer : inner_edge_wrapper->producers_) {
      //       producers.emplace_back(producer->node_);
      //     }
      //   } else {
      //     producers.emplace_back(producer_node);
      //   }
      // }
      findProducerNode(edge_wrapper, producers);
      for (auto node : producers) {
        oss << "p" << (void *)node << "->"
            << "p" << (void *)output;
        if (output->getName().empty()) {
          oss << "\n";
        } else {
          std::string label = "\"" + output->getName() + "\"";
          oss << "[label=" << label << "]\n";
        }
      }
    }
  }
  // dump node
  for (auto node_wrapper : node_repository_) {
    Node *node = node_wrapper->node_;
    if (node->getGraphFlag()) {
      Graph *graph = (Graph *)node;
      graph->dump(oss);
    } else {
      if (node->getName().empty()) {
        oss << "p" << (void *)node << "[label=node]\n";
      } else {
        std::string label = "\"" + node->getName() + "\"";
        oss << "p" << (void *)node << "[label=" << label << "]\n";
      }
    }
  }
  // dump edge
  for (auto edge_wrapper : edge_repository_) {
    std::vector<Node *> producers;
    // for (auto producer : edge_wrapper->producers_) {
    //   auto producer_node = producer->node_;
    //   if (producer_node->getGraphFlag()) {
    //     Graph *graph = (Graph *)producer_node;
    //     EdgeWrapper *inner_edge_wrapper =
    //         graph->getEdgeWrapper(edge_wrapper->edge_);
    //     if (inner_edge_wrapper == nullptr) {
    //       // NNDEPLOY_LOGE("edge_wrapper[%s] is null!\n",
    //       //               edge_wrapper->name_.c_str());
    //       continue;
    //     }
    //     for (auto producer : inner_edge_wrapper->producers_) {
    //       producers.emplace_back(producer->node_);
    //     }
    //   } else {
    //     producers.emplace_back(producer_node);
    //   }
    // }
    findProducerNode(edge_wrapper, producers);
    std::vector<Node *> consumers;
    // for (auto consumer : edge_wrapper->consumers_) {
    //   auto consumer_node = consumer->node_;
    //   if (consumer_node->getGraphFlag()) {
    //     Graph *graph = (Graph *)consumer_node;
    //     EdgeWrapper *inner_edge_wrapper =
    //         graph->getEdgeWrapper(edge_wrapper->edge_);
    //     if (inner_edge_wrapper == nullptr) {
    //       // NNDEPLOY_LOGE("edge_wrapper[%s] is null!\n",
    //       //               edge_wrapper->name_.c_str());
    //       continue;
    //     }
    //     for (auto consumer : inner_edge_wrapper->consumers_) {
    //       consumers.emplace_back(consumer->node_);
    //     }
    //   } else {
    //     consumers.emplace_back(consumer_node);
    //   }
    // }
    findConsumerNode(edge_wrapper, consumers);
    for (auto producer : producers) {
      for (auto consumer : consumers) {
        oss << "p" << (void *)producer << "->"
            << "p" << (void *)consumer;
        if (edge_wrapper->edge_->getName().empty()) {
          oss << "\n";
        } else {
          std::string label = "\"" + edge_wrapper->edge_->getName() + "\"";
          oss << "[label=" << label << "]\n";
        }
      }
    }
  }

  // end
  oss << "}\n";
  return status;
}

void Graph::setTraceFlag(bool flag) {
  for (auto node_wrapper : node_repository_) {
    node_wrapper->node_->setTraceFlag(flag);
  }
}

std::vector<Edge *> Graph::trace(std::vector<Edge *> inputs) {
  base::Status status = base::kStatusCodeOk;
  this->setTraceFlag(true);
  std::vector<Edge *> outputs = this->operator()(inputs);
  NNDEPLOY_LOGI("trace outputs size: %d.\n", outputs.size());
  status = this->init();
  if (status != base::kStatusCodeOk) {
    NNDEPLOY_LOGE("init failed!");
    return std::vector<Edge *>();
  }
  status = this->dump();
  if (status != base::kStatusCodeOk) {
    NNDEPLOY_LOGE("dump failed!");
    return std::vector<Edge *>();
  }
  traced_ = true;
  return outputs;
}

base::Status Graph::construct() {
  base::Status status = base::kStatusCodeOk;

  // NNDEPLOY_LOGE("NAME: %s start\n", name_.c_str());

  // NNDEPLOY_LOGI("###########################\n");
  // NNDEPLOY_LOGI("parallel_type_!\n");
  // NNDEPLOY_LOGI("###########################\n");
  // base::ParallelType parallel_type_ = parallel_type_;

  // NNDEPLOY_LOGI("###########################\n");
  // NNDEPLOY_LOGI("Parameter Validation Phase!\n");
  // NNDEPLOY_LOGI("###########################\n");
  for (auto node_wrapper : node_repository_) {
    NNDEPLOY_CHECK_PARAM_NULL_RET_STATUS(node_wrapper->node_,
                                         "edge_repository_ node is null!");
  }
  for (auto edge_wrapper : edge_repository_) {
    NNDEPLOY_CHECK_PARAM_NULL_RET_STATUS(edge_wrapper->edge_,
                                         "edge_repository_ edge is null!");
    if (edge_wrapper->producers_.empty() && edge_wrapper->consumers_.empty()) {
      NNDEPLOY_LOGI("graph[%s] this edge[%s] is useless!\n", name_.c_str(),
                    edge_wrapper->edge_->getName().c_str());
    }
  }

  // NNDEPLOY_LOGI("####################\n");
  // NNDEPLOY_LOGI("Mark Predecessors And Successors Phase!\n");
  // NNDEPLOY_LOGI("####################\n");
  for (auto node_wrapper : node_repository_) {
    Node *node = node_wrapper->node_;
    node->setDebugFlag(is_debug_);
    node->setTimeProfileFlag(is_time_profile_);
    node->setParallelType(parallel_type_);
    node->setInnerFlag(true);
    std::vector<Edge *> inputs = node->getAllInput();
    // NNDEPLOY_LOGE("NODE: %s.\n", node->getName().c_str());
    for (auto input : inputs) {
      EdgeWrapper *input_wrapper = findEdgeWrapper(edge_repository_, input);
      NNDEPLOY_CHECK_PARAM_NULL_RET_STATUS(input_wrapper,
                                           "input_wrapper is null!");
      // NNDEPLOY_LOGE("input_wrapper: %s.\n",
      //               input_wrapper->edge_->getName().c_str());
      for (auto producer : input_wrapper->producers_) {
        insertUnique(node_wrapper->predecessors_, producer);
        // NNDEPLOY_LOGE("producer: %s.\n", producer->node_->getName().c_str());
      }
    }
    std::vector<Edge *> outputs = node->getAllOutput();
    for (auto output : outputs) {
      EdgeWrapper *output_wrapper = findEdgeWrapper(edge_repository_, output);
      NNDEPLOY_CHECK_PARAM_NULL_RET_STATUS(output_wrapper,
                                           "output_wrapper is null!");
      // NNDEPLOY_LOGE("output_wrapper: %s.\n",
      //               output_wrapper->edge_->getName().c_str());
      for (auto consumer : output_wrapper->consumers_) {
        insertUnique(node_wrapper->successors_, consumer);
        // NNDEPLOY_LOGE("consumer: %s.\n", consumer->node_->getName().c_str());
      }
    }
  }

  // NNDEPLOY_LOGI("##############\n");
  // NNDEPLOY_LOGI("construct edge\n");
  // NNDEPLOY_LOGI("##############\n");
  for (auto edge_wrapper : edge_repository_) {
    // NNDEPLOY_LOGE("edge: %s, %p.\n", edge_wrapper->edge_->getName().c_str(),
    //               edge_wrapper->edge_);
    std::vector<Node *> producers;
    for (auto producer : edge_wrapper->producers_) {
      producers.emplace_back(producer->node_);
      // NNDEPLOY_LOGE("producer: %s.\n", producer->node_->getName().c_str());
    }
    std::vector<Node *> consumers;
    for (auto consumer : edge_wrapper->consumers_) {
      consumers.emplace_back(consumer->node_);
      // NNDEPLOY_LOGE("consumer: %s.\n", consumer->node_->getName().c_str());
    }
    base::Status status = edge_wrapper->edge_->setParallelType(parallel_type_);
    NNDEPLOY_RETURN_ON_NEQ(status, base::kStatusCodeOk,
                           "setParallelType failed!");
    // 必须在abstract_edge管理该字段
    status = edge_wrapper->edge_->increaseProducers(producers);
    NNDEPLOY_RETURN_ON_NEQ(status, base::kStatusCodeOk,
                           "increaseProducers failed!");
    status = edge_wrapper->edge_->increaseConsumers(consumers);
    NNDEPLOY_RETURN_ON_NEQ(status, base::kStatusCodeOk,
                           "increaseConsumers failed!");
    status = edge_wrapper->edge_->construct();
    NNDEPLOY_RETURN_ON_NEQ(status, base::kStatusCodeOk,
                           "construct edge failed!");
    status = edge_wrapper->edge_->setQueueMaxSize(queue_max_size_);
    NNDEPLOY_RETURN_ON_NEQ(status, base::kStatusCodeOk,
                           "setQueueMaxSize failed!");
  }

  // if (!is_inner_) {
  //   for (auto iter : outputs_) {
  //     iter->markGraphOutput();
  //   }
  // }

  if (!is_external_stream_ && stream_ == nullptr) {
    stream_ = device::createStream(device_type_);
  }
  // TODO: 是否需要延迟到executor阶段？
  if (is_graph_node_share_stream_ &&
      parallel_type_ != base::kParallelTypePipeline) {
    for (auto node_wrapper : node_repository_) {
      node_wrapper->node_->setStream(stream_);
    }
  }

  // 没有生产者的为输入边
  for (auto edge_wrapper : edge_repository_) {
    if (edge_wrapper->producers_.empty()) {
      auto it = std::find(inputs_.begin(), inputs_.end(), edge_wrapper->edge_);
      if (it == inputs_.end()) {
        inputs_.emplace_back(edge_wrapper->edge_);
      }
    }
  }

  // 没有消费者的为输出边
  for (auto edge_wrapper : edge_repository_) {
    if (edge_wrapper->consumers_.empty()) {
      auto it =
          std::find(outputs_.begin(), outputs_.end(), edge_wrapper->edge_);
      if (it == outputs_.end()) {
        outputs_.emplace_back(edge_wrapper->edge_);
      }
    }
  }

  if (!is_inner_) {
    for (auto iter : outputs_) {
      NNDEPLOY_LOGI("markGraphOutput: %s.\n", iter->getName().c_str());
      iter->markGraphOutput();
    }
  }

  // NNDEPLOY_LOGE("NAME: %s end\n", name_.c_str());

  return status;
}

base::Status Graph::executor() {
  // NNDEPLOY_LOGI("name: %s executor start.\n", name_.c_str());
  base::Status status = base::kStatusCodeOk;

  // NNDEPLOY_LOGI("###########################\n");
  // NNDEPLOY_LOGI("parallel_type_!\n");
  // NNDEPLOY_LOGI("###########################\n");
  // base::ParallelType parallel_type_ = parallel_type_;

  // NNDEPLOY_LOGI("##############\n");
  // NNDEPLOY_LOGI("create executor\n");
  // NNDEPLOY_LOGI("##############\n");
  if (parallel_type_ == base::kParallelTypeNone) {
    executor_ = std::make_shared<SequentialExecutor>();
  } else if (parallel_type_ == base::kParallelTypeSequential) {
    executor_ = std::make_shared<SequentialExecutor>();
  } else if (parallel_type_ == base::kParallelTypeTask) {
    executor_ = std::make_shared<ParallelTaskExecutor>();
  } else if (parallel_type_ == base::kParallelTypePipeline) {
    executor_ = std::make_shared<ParallelPipelineExecutor>();
  } else {
    NNDEPLOY_LOGE("parallel_type_ is invalid!\n");
    return base::kStatusCodeErrorInvalidValue;
  }
  NNDEPLOY_CHECK_PARAM_NULL_RET_STATUS(executor_, "Create executor failed!");

  executor_->setStream(stream_);

  // NNDEPLOY_LOGI("##############\n");
  // NNDEPLOY_LOGI("executor init\n");
  // NNDEPLOY_LOGI("1. Optimizer Graph V1!\n");
  // NNDEPLOY_LOGI("2. Device Verification Phase!\n");
  // NNDEPLOY_LOGI("3. Optimizer Graph V2!\n");
  // NNDEPLOY_LOGI("4. Memory Allocation Phase!\n");
  // NNDEPLOY_LOGI("5. Cost Calculations!\n");
  // NNDEPLOY_LOGI("##############\n");
  status = executor_->init(edge_repository_, node_repository_);
  NNDEPLOY_RETURN_ON_NEQ(status, base::kStatusCodeOk, "executor init failed!");

  // NNDEPLOY_LOGI("name: %s executor start.\n", name_.c_str());
  return status;
}

Node *Graph::createNode4Py(const NodeDesc &desc) { return createNode(desc); }

EdgeWrapper *Graph::getEdgeWrapper(Edge *edge) {
  return findEdgeWrapper(edge_repository_, edge);
}

EdgeWrapper *Graph::getEdgeWrapper(const std::string &name) {
  for (auto edge_wrapper : edge_repository_) {
    if (edge_wrapper->name_ == name) {
      return edge_wrapper;
    }
  }
  return nullptr;
}

NodeWrapper *Graph::getNodeWrapper(Node *node) {
  return findNodeWrapper(node_repository_, node);
}

NodeWrapper *Graph::getNodeWrapper(const std::string &name) {
  return findNodeWrapper(node_repository_, name);
}

base::Status Graph::serialize(
    rapidjson::Value &json,
    rapidjson::Document::AllocatorType &allocator) const {
  base::Status status = base::kStatusCodeOk;
  status = Node::serialize(json, allocator);
  if (status != base::kStatusCodeOk) {
    return status;
  }
  if (!is_inner_) {
    // 写入设备类型
    // std::string device_type_str = base::deviceTypeToString(device_type_);
    // json.AddMember("device_type_",
    //                rapidjson::Value(device_type_str.c_str(), allocator),
    //                allocator);
    // 写入是否为外部流
    json.AddMember("is_external_stream_", is_external_stream_, allocator);
    // 序列化并行类型
    // std::string parallel_type_str = base::parallelTypeToString(parallel_type_);
    // json.AddMember("parallel_type_",
    //                rapidjson::Value(parallel_type_str.c_str(), allocator),
    //                allocator);

    // 序列化其他布尔标志
    json.AddMember("is_inner_", is_inner_, allocator);
    json.AddMember("is_time_profile_", is_time_profile_, allocator);
    json.AddMember("is_debug_", is_debug_, allocator);
    // json.AddMember("is_graph_", is_graph_, allocator);

    // 写入节点类型
    // std::string node_type_str = nodeTypeToString(node_type_);
    // json.AddMember("node_type_",
    //                rapidjson::Value(node_type_str.c_str(), allocator),
    //                allocator);
    json.AddMember("is_graph_node_share_stream_", is_graph_node_share_stream_,
                   allocator);
    json.AddMember("queue_max_size_", queue_max_size_, allocator);
  }

  if (!node_repository_.empty()) {
    rapidjson::Value node_repository_array(rapidjson::kArrayType);
    for (auto node_wrapper : node_repository_) {
      rapidjson::Value node_json(rapidjson::kObjectType);
      node_wrapper->node_->serialize(node_json, allocator);
      node_repository_array.PushBack(node_json, allocator);
    }
    json.AddMember("node_repository_", node_repository_array, allocator);
  }

  return status;
}
base::Status Graph::deserialize(rapidjson::Value &json) {
  base::Status status = Node::deserialize(json);
  if (status != base::kStatusCodeOk) {
    return status;
  }

  if (json.HasMember("is_graph_node_share_stream_") &&
      json["is_graph_node_share_stream_"].IsBool()) {
    is_graph_node_share_stream_ = json["is_graph_node_share_stream_"].GetBool();
  }

  if (json.HasMember("queue_max_size_") && json["queue_max_size_"].IsInt()) {
    queue_max_size_ = json["queue_max_size_"].GetInt();
  }

  if (!is_inner_) {
    if (json.HasMember("inputs_") && json["inputs_"].IsArray()) {
      const rapidjson::Value &inputs = json["inputs_"];
      for (rapidjson::SizeType i = 0; i < inputs.Size(); i++) {
        if (inputs[i].IsObject()) {
          std::string input_name = inputs[i]["name_"].GetString();
          Edge *edge = this->getEdge(input_name);
          if (edge == nullptr) {
            edge = this->createEdge(input_name);
          }
          if (edge == nullptr) {
            NNDEPLOY_LOGE("create edge failed\n");
            return base::kStatusCodeErrorInvalidValue;
          }
          insertUnique(inputs_, edge);
        }
      }
    }
    if (json.HasMember("outputs_") && json["outputs_"].IsArray()) {
      const rapidjson::Value &outputs = json["outputs_"];
      for (rapidjson::SizeType i = 0; i < outputs.Size(); i++) {
        if (outputs[i].IsObject()) {
          std::string output_name = outputs[i]["name_"].GetString();
          Edge *edge = this->getEdge(output_name);
          if (edge == nullptr) {
            edge = this->createEdge(output_name);
          }
          if (edge == nullptr) {
            NNDEPLOY_LOGE("create edge failed\n");
            return base::kStatusCodeErrorInvalidValue;
          }
          insertUnique(outputs_, edge);
        }
      }
    }
  }

  if (json.HasMember("node_repository_") &&
      json["node_repository_"].IsArray()) {
    const rapidjson::Value &nodes = json["node_repository_"];
    for (rapidjson::SizeType i = 0; i < nodes.Size(); i++) {
      if (nodes[i].IsObject()) {
        NodeDesc node_desc;
        rapidjson::Value &node_json = const_cast<rapidjson::Value &>(nodes[i]);
        status = node_desc.deserialize(node_json);
        if (status != base::kStatusCodeOk) {
          return status;
        }
        Node *node = this->createNode(node_desc);
        if (node == nullptr) {
          NNDEPLOY_LOGE("create node failed\n");
          return base::kStatusCodeErrorInvalidValue;
        }
        base::Status status = node->deserialize(node_json);
        if (status != base::kStatusCodeOk) {
          NNDEPLOY_LOGE("deserialize node failed\n");
          return status;
        }
      }
    }
  }

  return base::kStatusCodeOk;
}

// to json file
base::Status Graph::loadJson(const std::string &path) {
  return Node::deserialize(path);
}
// from json file
base::Status Graph::saveJson(const std::string &path) {
  return Node::serialize(path);
}

REGISTER_NODE("nndeploy::dag::Graph", Graph);

std::map<std::string, createGraphFunc> &getGlobalGraphCreatorMap() {
  static std::once_flag once;
  static std::shared_ptr<std::map<std::string, createGraphFunc>> creators;
  std::call_once(once, []() {
    creators.reset(new std::map<std::string, createGraphFunc>);
  });
  return *creators;
}

Graph *createGraph(const std::string &name, base::InferenceType inference_type,
                   base::DeviceType device_type, Edge *input, Edge *output,
                   base::ModelType model_type, bool is_path,
                   std::vector<std::string> model_value) {
  Graph *temp = nullptr;
  auto &creater_map = getGlobalGraphCreatorMap();
  if (creater_map.count(name) > 0) {
    temp = creater_map[name](name, inference_type, device_type, input, output,
                             model_type, is_path, model_value);
  }
  return temp;
}

}  // namespace dag
}  // namespace nndeploy
