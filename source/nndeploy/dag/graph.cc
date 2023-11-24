
#include "nndeploy/dag/graph.h"

#include "nndeploy/base/common.h"
#include "nndeploy/base/glic_stl_include.h"
#include "nndeploy/base/log.h"
#include "nndeploy/base/macro.h"
#include "nndeploy/base/object.h"
#include "nndeploy/base/status.h"
#include "nndeploy/base/string.h"
#include "nndeploy/base/time_profiler.h"
#include "nndeploy/base/value.h"
#include "nndeploy/dag/edge.h"
#include "nndeploy/dag/node.h"
#include "nndeploy/device/buffer.h"
#include "nndeploy/device/buffer_pool.h"
#include "nndeploy/device/device.h"
#include "nndeploy/device/tensor.h"

namespace nndeploy {
namespace dag {

Graph::Graph(const std::string& name, Edge* input, Edge* output)
    : Node(name, input, output) {
  param_ = std::make_shared<GraphParam>();
  if (nullptr == addEdge(input)) {
    constructed_ = false;
    return;
  }
  if (nullptr == addEdge(output)) {
    constructed_ = false;
    return;
  }
  constructed_ = true;
}
Graph::Graph(const std::string& name, std::initializer_list<Edge*> inputs,
             std::initializer_list<Edge*> outputs)
    : Node(name, inputs, outputs) {
  param_ = std::make_shared<GraphParam>();
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
  topo_sort_node_.clear();
  node_repository_.clear();
  edge_repository_.clear();
}

Edge* Graph::createEdge(const std::string& name) {
  Edge* edge = new Edge(name);
  EdgeWrapper* edge_wrapper = new EdgeWrapper();
  edge_wrapper->is_external_ = false;
  edge_wrapper->edge_ = edge;
  edge_repository_.emplace_back(edge_wrapper);
  return edge;
}
EdgeWrapper* Graph::addEdge(Edge* edge) {
  base::Status status = base::kStatusCodeOk;
  NNDEPLOY_CHECK_PARAM_NULL_RET_NULL(edge, "edge is null!");
  EdgeWrapper* edge_wrapper = new EdgeWrapper();
  edge_wrapper->is_external_ = true;
  edge_wrapper->edge_ = edge;
  edge_repository_.emplace_back(edge_wrapper);
  return edge_wrapper;
}

// template <typename T>
// Node* Graph::createNode(const std::string& name, Edge* input,
//                            Edge* output) {
//   NNDEPLOY_CHECK_PARAM_NULL_RET_NULL(input, "input is null!");
//   NNDEPLOY_CHECK_PARAM_NULL_RET_NULL(output, "output is null!");
//   Node* node = dynamic_cast<Node*>(new T(name, input, output));
//   NodeWrapper* node_wrapper = new NodeWrapper();
//   node_wrapper->is_external_ = false;
//   node_wrapper->node_ = node;
//   node_wrapper->name_ = name;
//   if (findEdgeWrapper(input) == nullptr) {
//     this->addEdge(input);
//   }
//   findEdgeWrapper(input)->consumers_.emplace_back(node_wrapper);
//   if (findEdgeWrapper(output) == nullptr) {
//     this->addEdge(output);
//   }
//   findEdgeWrapper(output)->producers_.emplace_back(node_wrapper);
//   node_repository_.emplace_back(node_wrapper);
//   return node;
// }
// template <typename T>
// Node* Graph::createNode(const std::string& name, std::vector<Edge*>
// inputs,
//                            std::vector<Edge*> outputs) {
//   if (inputs.empty() || outputs.empty()) {
//     NNDEPLOY_LOGE("inputs or outputs is empty!\n");
//     return nullptr;
//   }
//   Node* node = dynamic_cast<Node*>(new T(name, inputs, outputs));
//   NodeWrapper* node_wrapper = new NodeWrapper();
//   node_wrapper->is_external_ = false;
//   node_wrapper->node_ = node;
//   node_wrapper->name_ = name;
//   for (auto input : inputs) {
//     if (findEdgeWrapper(input) == nullptr) {
//       this->addEdge(input);
//     }
//     findEdgeWrapper(input)->consumers_.emplace_back(node_wrapper);
//   }
//   for (auto output : outputs) {
//     if (findEdgeWrapper(output) == nullptr) {
//       this->addEdge(output);
//     }
//     findEdgeWrapper(output)->producers_.emplace_back(node_wrapper);
//   }
//   node_repository_.emplace_back(node_wrapper);
//   return node;
// }
// template <typename T>
// Node* Graph::createInfer(const std::string& name, base::InferenceType
// type,
//                             Edge* input, Edge* output) {
//   NNDEPLOY_CHECK_PARAM_NULL_RET_NULL(input, "input is null!");
//   NNDEPLOY_CHECK_PARAM_NULL_RET_NULL(output, "output is null!");
//   Node* node = dynamic_cast<Node*>(new T(name, type, input, output));
//   NodeWrapper* node_wrapper = new NodeWrapper();
//   node_wrapper->is_external_ = false;
//   node_wrapper->node_ = node;
//   node_wrapper->name_ = name;
//   if (findEdgeWrapper(input) == nullptr) {
//     this->addEdge(input);
//   }
//   findEdgeWrapper(input)->consumers_.emplace_back(node_wrapper);
//   if (findEdgeWrapper(output) == nullptr) {
//     this->addEdge(output);
//   }
//   findEdgeWrapper(output)->producers_.emplace_back(node_wrapper);
//   node_repository_.emplace_back(node_wrapper);
//   return node;
// }
base::Status Graph::addNode(Node* node) {
  base::Status status = base::kStatusCodeOk;
  NNDEPLOY_CHECK_PARAM_NULL_RET_STATUS(node, "node is null!");
  NodeWrapper* node_wrapper = new NodeWrapper();
  node_wrapper->is_external_ = true;
  node_wrapper->node_ = node;
  node_wrapper->name_ = node->getName();
  for (auto input : node->getAllInput()) {
    EdgeWrapper* input_wrapper = findEdgeWrapper(input);
    if (findEdgeWrapper(input) == nullptr) {
      input_wrapper = this->addEdge(input);
    }
    input_wrapper->consumers_.emplace_back(node_wrapper);
  }
  for (auto output : node->getAllOutput()) {
    EdgeWrapper* output_wrapper = findEdgeWrapper(output);
    if (output_wrapper == nullptr) {
      output_wrapper = this->addEdge(output);
    }
    output_wrapper->producers_.emplace_back(node_wrapper);
  }

  node_repository_.emplace_back(node_wrapper);
  return status;
}
Node* Graph::getNode(const std::string& node_name) {
  for (auto node_wrapper : node_repository_) {
    if (node_wrapper->name_ == node_name) {
      return node_wrapper->node_;
    }
  }
  return nullptr;
}

base::Status Graph::setNodeParam(const std::string& node_name,
                                 base::Param* param) {
  base::Status status = base::kStatusCodeOk;
  NNDEPLOY_CHECK_PARAM_NULL_RET_STATUS(param, "param is null!");
  NodeWrapper* node_wrapper = findNodeWrapper(node_name);
  NNDEPLOY_CHECK_PARAM_NULL_RET_STATUS(node_wrapper, "node_wrapper is null!");
  status = node_wrapper->node_->setParam(param);
  return status;
}

base::Param* Graph::getNodeParam(const std::string& node_name) {
  NodeWrapper* node_wrapper = findNodeWrapper(node_name);
  NNDEPLOY_CHECK_PARAM_NULL_RET_NULL(node_wrapper, "node_wrapper is null!");
  return node_wrapper->node_->getParam();
}

void Graph::setPipelineParallel(bool is_pipeline_parallel) {
  Node::setPipelineParallel(is_pipeline_parallel);
  for (auto node_wrapper : node_repository_) {
    node_wrapper->node_->setPipelineParallel(is_pipeline_parallel);
  }
}

base::Status Graph::init() {
  base::Status status = base::kStatusCodeOk;

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
  }

  // NNDEPLOY_LOGI("####################\n");
  // NNDEPLOY_LOGI("Mark Predecessors And Successors Phase!\n");
  // NNDEPLOY_LOGI("####################\n");
  for (auto node_wrapper : node_repository_) {
    Node* node = node_wrapper->node_;
    std::vector<Edge*> inputs = node->getAllInput();
    for (auto input : inputs) {
      EdgeWrapper* input_wrapper = findEdgeWrapper(input);
      NNDEPLOY_CHECK_PARAM_NULL_RET_STATUS(input_wrapper,
                                           "input_wrapper is null!");
      node_wrapper->predecessors_.assign(input_wrapper->producers_.begin(),
                                         input_wrapper->producers_.end());
    }
    std::vector<Edge*> outputs = node->getAllOutput();
    for (auto output : outputs) {
      EdgeWrapper* output_wrapper = findEdgeWrapper(output);
      NNDEPLOY_CHECK_PARAM_NULL_RET_STATUS(output_wrapper,
                                           "output_wrapper is null!");
      node_wrapper->successors_.assign(output_wrapper->consumers_.begin(),
                                       output_wrapper->consumers_.end());
    }
  }

  // NNDEPLOY_LOGI("##############\n");
  // NNDEPLOY_LOGI("TopologicalSort and Check Cycle!\n");
  // NNDEPLOY_LOGI("##############\n");
  /**
   * @brief
   * @note
   * # 联通图（多个独立的子图）
   * # node并行（图中存在可以并行的node）
   * # 流水线并行（通过流水线的方式并行）
   * # 条件并行（通过条件判断的方式并行）
   */
  status = topologicalSort();
  if (status != base::kStatusCodeOk) {
    NNDEPLOY_LOGE("Toposort failed");
    return status;
  }

  // // NNDEPLOY_LOGI("############################\n");
  // // NNDEPLOY_LOGI("Checking for Unvisited Edge!\n");
  // // NNDEPLOY_LOGI("############################\n");

  // // NNDEPLOY_LOGI("############################\n");
  // // NNDEPLOY_LOGI("Optimizer Graph V1!\n");
  // // NNDEPLOY_LOGI("############################\n");

  // // NNDEPLOY_LOGI("#########################\n");
  // // NNDEPLOY_LOGI("Device Verification Phase!\n");
  // // NNDEPLOY_LOGI("#########################\n");

  // // NNDEPLOY_LOGI("############################\n");
  // // NNDEPLOY_LOGI("Optimizer Graph V2!\n");
  // // NNDEPLOY_LOGI("############################\n");

  // NNDEPLOY_LOGI("#######################\n");
  // NNDEPLOY_LOGI("Node Initialize Phase!\n");
  // NNDEPLOY_LOGI("#######################\n");
  for (auto node_vec : topo_sort_node_) {
    for (auto node : node_vec) {
      status = node->init();
      if (status != base::kStatusCodeOk) {
        NNDEPLOY_LOGE("Node init failed!\n");
        return status;
      }
    }
  }

  // // NNDEPLOY_LOGI("########################\n");
  // // NNDEPLOY_LOGI("Memory Allocation Phase!\n");
  // // NNDEPLOY_LOGI("########################\n");

  // // NNDEPLOY_LOGI("#######################\n");
  // // NNDEPLOY_LOGI("Cost Calculations!\n");
  // // NNDEPLOY_LOGI("#######################\n");

  return status;
}

base::Status Graph::deinit() {
  base::Status status = base::kStatusCodeOk;
  // NNDEPLOY_LOGI("#######################\n");
  // NNDEPLOY_LOGI("Node DeInitialize Phase!\n");
  // NNDEPLOY_LOGI("#######################\n");
  for (auto node_vec : topo_sort_node_) {
    for (auto node : node_vec) {
      status = node->deinit();
      if (status != base::kStatusCodeOk) {
        NNDEPLOY_LOGE("Node deinit failed!\n");
        return status;
      }
    }
  }
  return status;
}

// base::Status Graph::reshape() {
//   base::Status status = base::kStatusCodeOk;
//   // NNDEPLOY_LOGI("#######################\n");
//   // NNDEPLOY_LOGI("Node reshape Phase!\n");
//   // NNDEPLOY_LOGI("#######################\n");
//   for (auto node_vec : topo_sort_node_) {
//     for (auto node : node_vec) {
//       status = node->reshape();
//       if (status != base::kStatusCodeOk) {
//         NNDEPLOY_LOGE("Node run failed!\n");
//         return status;
//       }
//     }
//   }
//   return status;
// }

base::Status Graph::run() {
  base::Status status = base::kStatusCodeOk;
  is_running_ = true;
  // NNDEPLOY_LOGI("#######################\n");
  // NNDEPLOY_LOGI("Node run Phase!\n");
  // NNDEPLOY_LOGI("#######################\n");
  for (auto node_vec : topo_sort_node_) {
    for (auto node : node_vec) {
      NNDEPLOY_TIME_POINT_START(node->getName());
      NNDEPLOY_LOGE("NODE RUN %s\n", node->getName().c_str());
      status = node->run();
      NNDEPLOY_TIME_POINT_END(node->getName());
      if (status != base::kStatusCodeOk) {
        NNDEPLOY_LOGE("Node run failed!\n");
        return status;
      }
    }
  }
  return status;
}

base::Status Graph::dump(std::ostream& oss) {
  base::Status status = base::kStatusCodeOk;
  // NNDEPLOY_LOGI("#######################\n");
  // NNDEPLOY_LOGI("Node dump Phase!\n");
  // NNDEPLOY_LOGI("#######################\n");
  if (name_.empty()) {
    oss << "digraph graph {\n";
  } else {
    oss << "digraph " << name_ << " {\n";
  }
  for (auto node_vec : topo_sort_node_) {
    for (auto node : node_vec) {
      NodeWrapper* node_wrapper = findNodeWrapper(node);
      if (node_wrapper->predecessors_.empty()) {
        auto inputs = node->getAllInput();
        for (auto input : inputs) {
          oss << "p" << (void*)input << "[label=input]\n";
          oss << "p" << (void*)input << "->"
              << "p" << (void*)node;
          if (input->getName().empty()) {
            oss << "\n";
          } else {
            oss << "[label=" << input->getName() << "]\n";
          }
        }
      }
      if (node->getName().empty()) {
        oss << "p" << (void*)node << "\n";
      } else {
        oss << "p" << (void*)node << "[label=" << node->getName() << "]\n";
      }
      if (node_wrapper->successors_.empty()) {
        auto outputs = node->getAllOutput();
        for (auto output : outputs) {
          oss << "p" << (void*)output << "[label=output]\n";
          oss << "p" << (void*)node << "->"
              << "p" << (void*)output;
          if (output->getName().empty()) {
            oss << "\n";
          } else {
            oss << "[label=" << output->getName() << "]\n";
          }
        }
      } else {
        for (auto successor : node_wrapper->successors_) {
          oss << "p" << (void*)node << "->"
              << "p" << (void*)(successor->node_);
          auto outputs = node->getAllOutput();
          auto inputs = successor->node_->getAllInput();
          Edge* out_in = nullptr;
          for (auto output : outputs) {
            for (auto input : inputs) {
              if (output == input) {
                out_in = output;
              }
            }
          }
          if (out_in != nullptr) {
            if (out_in->getName().empty()) {
              oss << "\n";
            } else {
              oss << "[label=" << out_in->getName() << "]\n";
            }
          }
        }
      }
    }
  }
  oss << "}\n";
  return status;
}

EdgeWrapper* Graph::findEdgeWrapper(Edge* edge) {
  for (auto edge_wrapper : edge_repository_) {
    if (edge_wrapper->edge_ == edge) {
      return edge_wrapper;
    }
  }
  return nullptr;
}
NodeWrapper* Graph::findNodeWrapper(const std::string& node_name) {
  for (auto node_wrapper : node_repository_) {
    if (node_wrapper->name_ == node_name) {
      return node_wrapper;
    }
  }
  return nullptr;
}
NodeWrapper* Graph::findNodeWrapper(Node* node) {
  for (auto node_wrapper : node_repository_) {
    if (node_wrapper->node_ == node) {
      return node_wrapper;
    }
  }
  return nullptr;
}

std::vector<NodeWrapper*> Graph::findStartNodes() {
  std::vector<NodeWrapper*> start_nodes;
  for (auto node_wrapper : node_repository_) {
    if (node_wrapper->predecessors_.empty()) {
      start_nodes.emplace_back(node_wrapper);
    }
  }
  return start_nodes;
}

std::vector<NodeWrapper*> Graph::findEndNodes() {
  std::vector<NodeWrapper*> end_nodes;
  for (auto node_wrapper : node_repository_) {
    if (node_wrapper->successors_.empty()) {
      end_nodes.emplace_back(node_wrapper);
    }
  }
  return end_nodes;
}

base::Status Graph::TopoSortBFS(NodeWrapper* node_wrapper) {
  std::vector<Node*> dst;
  node_wrapper->color_ = kNodeColorGray;
  std::deque<NodeWrapper*> node_deque;
  node_deque.emplace_back(node_wrapper);
  while (!node_deque.empty()) {
    NodeWrapper* node_wrapper = node_deque.front();
    if (node_wrapper->color_ == kNodeColorBlack) {
      node_deque.pop_front();
      continue;
    }
    bool flag = false;
    for (auto predecessor : node_wrapper->predecessors_) {
      if (predecessor->color_ != kNodeColorBlack) {
        predecessor->color_ = kNodeColorGray;
        node_deque.emplace_front(predecessor);
        flag = true;
        break;
      }
    }
    if (flag) {
      continue;
    }
    for (auto successor : node_wrapper->successors_) {
      if (successor->color_ == kNodeColorBlack) {
        NNDEPLOY_LOGE("Cycle detected in graph");
        return base::kStatusCodeErrorInvalidValue;
      } else if (successor->color_ == kNodeColorWhite) {
        successor->color_ = kNodeColorGray;
        node_deque.emplace_back(successor);
      }
    }
    node_deque.pop_front();
    node_wrapper->color_ = kNodeColorBlack;
    dst.emplace_back(node_wrapper->node_);
  }
  topo_sort_node_.emplace_back(dst);
  return base::kStatusCodeOk;
}

base::Status Graph::TopoSortDFS(NodeWrapper* node_wrapper,
                                std::stack<NodeWrapper*>& dst) {
  base::Status status = base::kStatusCodeOk;
  node_wrapper->color_ = kNodeColorGray;
  for (auto successor : node_wrapper->successors_) {
    if (successor->color_ == kNodeColorWhite) {
      status = TopoSortDFS(successor, dst);
    } else if (successor->color_ == kNodeColorGray) {
      NNDEPLOY_LOGE("Cycle detected in graph");
      status = base::kStatusCodeErrorInvalidValue;
    } else {
      continue;
    }
  }
  if (status != base::kStatusCodeOk) {
    return status;
  }
  node_wrapper->color_ = kNodeColorBlack;
  dst.push(node_wrapper);
  return base::kStatusCodeOk;
}

/**
 * @brief topo sort and check cycle
 *
 * @return base::Status
 */
base::Status Graph::topologicalSort() {
  base::Status status = base::kStatusCodeOk;

  std::vector<NodeWrapper*> start_nodes = findStartNodes();
  if (start_nodes.empty()) {
    NNDEPLOY_LOGE("No start node found in graph");
    return base::kStatusCodeErrorInvalidValue;
  }
  GraphParam* param = dynamic_cast<GraphParam*>(this->param_.get());
  if (param->topo_sort_type_ == kTopoSortTypeBFS) {
    for (auto node_wrapper : start_nodes) {
      if (node_wrapper->color_ == kNodeColorBlack) {
        continue;
      }
      status = TopoSortBFS(node_wrapper);
      if (status != base::kStatusCodeOk) {
        NNDEPLOY_LOGE("TopoSortBFS failed");
        return status;
      }
    }
  } else {
    std::stack<NodeWrapper*> dst;
    for (auto node_wrapper : start_nodes) {
      if (node_wrapper->color_ == kNodeColorBlack) {
        continue;
      }
      status = TopoSortDFS(node_wrapper, dst);
      if (status != base::kStatusCodeOk) {
        NNDEPLOY_LOGE("TopoSortDFS failed");
        return status;
      }
    }
    std::vector<Node*> node_dst;
    while (!dst.empty()) {
      node_dst.emplace_back(dst.top()->node_);
      dst.pop();
    }
    topo_sort_node_.emplace_back(node_dst);
  }

  return status;
}

std::map<std::string, createGraphFunc>& getGlobalGraphCreatorMap() {
  static std::once_flag once;
  static std::shared_ptr<std::map<std::string, createGraphFunc>> creators;
  std::call_once(once, []() {
    creators.reset(new std::map<std::string, createGraphFunc>);
  });
  return *creators;
}

Graph* createGraph(const std::string& name, base::InferenceType inference_type,
                   base::DeviceType device_type, Edge* input, Edge* output,
                   base::ModelType model_type, bool is_path,
                   std::vector<std::string> model_value) {
  Graph* temp = nullptr;
  auto& creater_map = getGlobalGraphCreatorMap();
  if (creater_map.count(name) > 0) {
    temp = creater_map[name](name, inference_type, device_type, input, output,
                             model_type, is_path, model_value);
  }
  return temp;
}

}  // namespace dag
}  // namespace nndeploy
