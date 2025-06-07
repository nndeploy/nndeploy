
#include "nndeploy/dag/composite_node.h"

namespace nndeploy {
namespace dag {

CompositeNode::~CompositeNode() {
  for (auto edge_wrapper : edge_repository_) {
    if (!edge_wrapper->is_external_) {
      delete edge_wrapper->edge_;
    }
    delete edge_wrapper;
  }
  edge_repository_.clear();
  for (auto node_wrapper : node_repository_) {
    if (!node_wrapper->is_external_) {
      delete node_wrapper->node_;
    }
    delete node_wrapper;
  }
}

base::Status CompositeNode::init() {
  base::Status status = base::kStatusCodeOk;
  status = construct();
  if (status != base::kStatusCodeOk) {
    NNDEPLOY_LOGI("construct failed!");
  }
  for (auto node_wrapper : node_repository_) {
    if (node_wrapper->node_->getInitialized()) {
      continue;
    }
    status = node_wrapper->node_->init();
    NNDEPLOY_RETURN_ON_NEQ(status, base::kStatusCodeOk, "node init failed!");
    node_wrapper->node_->setInitializedFlag(true);
  }
  return status;
}

base::Status CompositeNode::deinit() {
  base::Status status = base::kStatusCodeOk;
  for (auto node_wrapper : node_repository_) {
    if (node_wrapper->node_->getInitialized()) {
      status = node_wrapper->node_->deinit();
      NNDEPLOY_RETURN_ON_NEQ(status, base::kStatusCodeOk,
                             "node deinit failed!");
      node_wrapper->node_->setInitializedFlag(false);
    }
  }
  return status;
}

Edge *CompositeNode::findEdgeByName(const std::vector<Edge *> &edges,
                                    const std::string &name) const {
  for (auto *edge : edges) {
    if (edge && edge->getName() == name) {
      return edge;
    }
  }
  return nullptr;
}

Edge *CompositeNode::getEdge(const std::string &name) {
  for (EdgeWrapper *edge_wrapper : edge_repository_) {
    if (edge_wrapper->name_ == name) {
      return edge_wrapper->edge_;
    }
  }
  return nullptr;
}

Edge *CompositeNode::createEdge(const std::string &name) {
  std::string unique_name = name;
  if (unique_name.empty()) {
    unique_name = "edge_" + base::getUniqueString();
  }
  Edge *edge = new Edge(unique_name);
  EdgeWrapper *edge_wrapper = new EdgeWrapper();
  edge_wrapper->is_external_ = false;
  edge_wrapper->edge_ = edge;
  edge_wrapper->name_ = unique_name;
  edge_repository_.emplace_back(edge_wrapper);
  return edge;
}

Node *CompositeNode::createNode(const NodeDesc &desc) {
  const std::string &name = desc.getName();
  const std::string &node_key = desc.getKey();
  std::vector<std::string> input_names = desc.getInputs();
  std::vector<std::string> output_names = desc.getOutputs();

  std::vector<Edge *> composite_inputs = getAllInput();
  std::vector<Edge *> composite_outputs = getAllOutput();

  std::vector<Edge *> inputs;
  for (auto input_name : input_names) {
    Edge *input = findEdgeByName(composite_inputs, input_name);
    if (!input) {
      input = getEdge(input_name);
      if (!input) {
        input = createEdge(input_name);
      }
    }
    inputs.emplace_back(input);
  }
  std::vector<Edge *> outputs;
  for (auto output_name : output_names) {
    Edge *output = findEdgeByName(composite_outputs, output_name);
    if (!output) {
      output = getEdge(output_name);
      if (!output) {
        output = createEdge(output_name);
      }
    }
    outputs.emplace_back(output);
  }
  Node *node = nndeploy::dag::createNode(node_key, name, inputs, outputs);
  if (node == nullptr) {
    NNDEPLOY_LOGE("create node[%s] failed!\n", name.c_str());
    return nullptr;
  }
  NodeWrapper *node_wrapper = new NodeWrapper();
  node_wrapper->is_external_ = false;
  node_wrapper->node_ = node;
  node_wrapper->name_ = name;

  Graph *graph = getGraph();
  for (auto input_name : input_names) {
    Edge *input = findEdgeByName(composite_inputs, input_name);
    if (input != nullptr) {
      EdgeWrapper *edge_wrapper = graph->getEdgeWrapper(input);
      edge_wrapper->consumers_.emplace_back(node_wrapper);
    }
  }
  for (auto output_name : output_names) {
    Edge *output = findEdgeByName(composite_outputs, output_name);
    if (output != nullptr) {
      EdgeWrapper *edge_wrapper = graph->getEdgeWrapper(output);
      edge_wrapper->producers_.emplace_back(node_wrapper);
    }
  }

  node_repository_.emplace_back(node_wrapper);

  return node;
}

base::Status CompositeNode::construct() {
  base::Status status = base::kStatusCodeOk;

  // NNDEPLOY_LOGE("NAME: %s start\n", name_.c_str());

  // NNDEPLOY_LOGI("###########################\n");
  // NNDEPLOY_LOGI("Parameter Validation Phase!\n");
  // NNDEPLOY_LOGI("###########################\n");
  for (auto node_wrapper : node_repository_) {
    NNDEPLOY_CHECK_PARAM_NULL_RET_STATUS(node_wrapper->node_,
                                         "edge_repository_ node is null!");
  }
  // for (auto edge_wrapper : edge_repository_) {
  //   NNDEPLOY_CHECK_PARAM_NULL_RET_STATUS(edge_wrapper->edge_,
  //                                        "edge_repository_ edge is null!");
  //   if (edge_wrapper->producers_.empty() && edge_wrapper->consumers_.empty())
  //   {
  //     NNDEPLOY_LOGI("graph[%s] this edge[%s] is useless!\n", name_.c_str(),
  //                   edge_wrapper->edge_->getName().c_str());
  //   }
  // }

  // NNDEPLOY_LOGI("####################\n");
  // NNDEPLOY_LOGI("Mark Predecessors And Successors Phase!\n");
  // NNDEPLOY_LOGI("####################\n");
  for (auto node_wrapper : node_repository_) {
    Node *node = node_wrapper->node_;
    node->setDebugFlag(is_debug_);
    node->setTimeProfileFlag(is_time_profile_);
    node->setParallelType(base::kParallelTypeSequential);
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
    base::Status status =
        edge_wrapper->edge_->setParallelType(base::kParallelTypeSequential);
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
    // status = edge_wrapper->edge_->setQueueMaxSize(queue_max_size_);
    // NNDEPLOY_RETURN_ON_NEQ(status, base::kStatusCodeOk,
    //                        "setQueueMaxSize failed!");
  }

  if (!is_external_stream_ && stream_ == nullptr) {
    stream_ = device::createStream(device_type_);
  }
  // TODO: 是否需要延迟到executor阶段？
  if (parallel_type_ != base::kParallelTypePipeline) {
    for (auto node_wrapper : node_repository_) {
      node_wrapper->node_->setStream(stream_);
    }
  }

  // // 没有生产者的为输入边
  // for (auto edge_wrapper : edge_repository_) {
  //   if (edge_wrapper->producers_.empty()) {
  //     auto it = std::find(inputs_.begin(), inputs_.end(),
  //     edge_wrapper->edge_); if (it == inputs_.end()) {
  //       inputs_.emplace_back(edge_wrapper->edge_);
  //     }
  //   }
  // }

  // // 没有消费者的为输出边
  // for (auto edge_wrapper : edge_repository_) {
  //   if (edge_wrapper->consumers_.empty()) {
  //     auto it =
  //         std::find(outputs_.begin(), outputs_.end(), edge_wrapper->edge_);
  //     if (it == outputs_.end()) {
  //       outputs_.emplace_back(edge_wrapper->edge_);
  //     }
  //   }
  // }

  // NNDEPLOY_LOGE("NAME: %s end\n", name_.c_str());

  return status;
}

std::vector<NodeWrapper *> CompositeNode::sortDFS() {
  std::vector<NodeWrapper *> topo_sort_node;
  topoSortDFS(node_repository_, topo_sort_node);
  return topo_sort_node;
}

base::Status CompositeNode::serialize(
    rapidjson::Value &json,
    rapidjson::Document::AllocatorType &allocator) {
  base::Status status = base::kStatusCodeOk;
  status = Node::serialize(json, allocator);
  if (status != base::kStatusCodeOk) {
    NNDEPLOY_LOGE("serialize node failed\n");
    return status;
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

base::Status CompositeNode::deserialize(rapidjson::Value &json) {
  base::Status status = Node::deserialize(json);
  if (status != base::kStatusCodeOk) {
    NNDEPLOY_LOGE("deserialize node failed\n");
    return status;
  }
  // if (json.HasMember("inputs_") && json["inputs_"].IsArray()) {
  //   const rapidjson::Value &inputs = json["inputs_"];
  //   for (rapidjson::SizeType i = 0; i < inputs.Size(); i++) {
  //     if (inputs[i].IsObject()) {
  //       std::string input_name = inputs[i]["name_"].GetString();
  //       Edge *edge = this->getEdge(input_name);
  //       if (edge == nullptr) {
  //         edge = this->createEdge(input_name);
  //       }
  //       if (edge == nullptr) {
  //         NNDEPLOY_LOGE("create edge failed\n");
  //         return base::kStatusCodeErrorInvalidValue;
  //       }
  //       insertUnique(inputs_, edge);
  //     }
  //   }
  // }
  // if (json.HasMember("outputs_") && json["outputs_"].IsArray()) {
  //   const rapidjson::Value &outputs = json["outputs_"];
  //   for (rapidjson::SizeType i = 0; i < outputs.Size(); i++) {
  //     if (outputs[i].IsObject()) {
  //       std::string output_name = outputs[i]["name_"].GetString();
  //       Edge *edge = this->getEdge(output_name);
  //       if (edge == nullptr) {
  //         edge = this->createEdge(output_name);
  //       }
  //       if (edge == nullptr) {
  //         NNDEPLOY_LOGE("create edge failed\n");
  //         return base::kStatusCodeErrorInvalidValue;
  //       }
  //       insertUnique(outputs_, edge);
  //     }
  //   }
  // }
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
  return status;
}

}  // namespace dag
}  // namespace nndeploy
