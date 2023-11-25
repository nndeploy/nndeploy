
#include "nndeploy/dag/graph/executor.h"

namespace nndeploy {
namespace dag {

Edge* getEdge(std::vector<EdgeWrapper*>& edge_repository,
              const std::string& edge_name) {
  for (auto edge_wrapper : edge_repository) {
    if (edge_wrapper->name_ == edge_name) {
      return edge_wrapper->edge_;
    }
  }
  return nullptr;
}
EdgeWrapper* findEdgeWrapper(std::vector<EdgeWrapper*>& edge_repository,
                             const std::string& edge_name) {
  for (auto edge_wrapper : edge_repository) {
    if (edge_wrapper->name_ == edge_name) {
      return edge_wrapper;
    }
  }
  return nullptr;
}
EdgeWrapper* findEdgeWrapper(std::vector<EdgeWrapper*>& edge_repository,
                             Edge* edge) {
  for (auto edge_wrapper : edge_repository) {
    if (edge_wrapper->edge_ == edge) {
      return edge_wrapper;
    }
  }
  return nullptr;
}
std::vector<EdgeWrapper*> findStartEdges(
    std::vector<EdgeWrapper*>& edge_repository) {
  std::vector<EdgeWrapper*> start_edge;
  for (auto edge_wrapper : edge_repository) {
    if (edge_wrapper->producers_.empty()) {
      start_edge.emplace_back(edge_wrapper);
    }
  }
  return start_edge;
}
std::vector<EdgeWrapper*> findEndEdges(
    std::vector<EdgeWrapper*>& edge_repository) {
  std::vector<EdgeWrapper*> end_edge;
  for (auto edge_wrapper : edge_repository) {
    if (edge_wrapper->consumers_.empty()) {
      end_edge.emplace_back(edge_wrapper);
    }
  }
  return end_edge;
}

Node* getNode(std::vector<NodeWrapper*>& node_repository,
              const std::string& node_name) {
  for (auto node_wrapper : node_repository) {
    if (node_wrapper->name_ == node_name) {
      return node_wrapper->node_;
    }
  }
  return nullptr;
}
NodeWrapper* findNodeWrapper(std::vector<NodeWrapper*>& node_repository,
                             const std::string& node_name) {
  for (auto node_wrapper : node_repository) {
    if (node_wrapper->name_ == node_name) {
      return node_wrapper;
    }
  }
  return nullptr;
}
NodeWrapper* findNodeWrapper(std::vector<NodeWrapper*>& node_repository,
                             Node* node) {
  for (auto node_wrapper : node_repository) {
    if (node_wrapper->node_ == node) {
      return node_wrapper;
    }
  }
  return nullptr;
}
std::vector<NodeWrapper*> findStartNodes(
    std::vector<NodeWrapper*>& node_repository) {
  std::vector<NodeWrapper*> start_nodes;
  for (auto node_wrapper : node_repository) {
    if (node_wrapper->predecessors_.empty()) {
      start_nodes.emplace_back(node_wrapper);
    }
  }
  return start_nodes;
}
std::vector<NodeWrapper*> findEndNodes(
    std::vector<NodeWrapper*>& node_repository) {
  std::vector<NodeWrapper*> end_nodes;
  for (auto node_wrapper : node_repository) {
    if (node_wrapper->successors_.empty()) {
      end_nodes.emplace_back(node_wrapper);
    }
  }
  return end_nodes;
}

base::Status dump(std::vector<NodeWrapper*>& node_repository,
                  const std::string& name, std::ostream& oss) {
  base::Status status = base::kStatusCodeOk;
  // NNDEPLOY_LOGI("#######################\n");
  // NNDEPLOY_LOGI("Node dump Phase!\n");
  // NNDEPLOY_LOGI("#######################\n");
  if (name.empty()) {
    oss << "digraph graph {\n";
  } else {
    oss << "digraph " << name << " {\n";
  }
  for (auto node_wrapper : node_repository) {
    Node* node = node_wrapper->node_;
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
  oss << "}\n";
  return status;
}

base::Status topoSortBFS(
    std::vector<NodeWrapper*>& node_repository,
    std::vector<std::vector<NodeWrapper*>>& topo_sort_node) {
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

base::Status topoSortDFS(
    std::vector<NodeWrapper*>& node_repository,
    std::vector<std::vector<NodeWrapper*>>& topo_sort_node) {
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

base::Status topoSort(std::vector<NodeWrapper*>& node_repository,
                      TopoSortType topo_sort_type,
                      std::vector<std::vector<NodeWrapper*>>& topo_sort_node) {}

}  // namespace dag
}  // namespace nndeploy