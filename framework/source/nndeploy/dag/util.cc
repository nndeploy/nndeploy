
#include "nndeploy/dag/util.h"

namespace nndeploy {
namespace dag {

Edge *getEdge(std::vector<EdgeWrapper *> &edge_repository,
              const std::string &edge_name) {
  for (auto edge_wrapper : edge_repository) {
    if (edge_wrapper->name_ == edge_name) {
      return edge_wrapper->edge_;
    }
  }
  return nullptr;
}
EdgeWrapper *findEdgeWrapper(std::vector<EdgeWrapper *> &edge_repository,
                             const std::string &edge_name) {
  for (auto edge_wrapper : edge_repository) {
    if (edge_wrapper->name_ == edge_name) {
      return edge_wrapper;
    }
  }
  return nullptr;
}
EdgeWrapper *findEdgeWrapper(std::vector<EdgeWrapper *> &edge_repository,
                             Edge *edge) {
  for (auto edge_wrapper : edge_repository) {
    if (edge_wrapper->edge_ == edge) {
      return edge_wrapper;
    }
  }
  return nullptr;
}
std::vector<EdgeWrapper *> findStartEdges(
    std::vector<EdgeWrapper *> &edge_repository) {
  std::vector<EdgeWrapper *> start_edge;
  for (auto edge_wrapper : edge_repository) {
    if (edge_wrapper->producers_.empty()) {
      start_edge.emplace_back(edge_wrapper);
    }
  }
  return start_edge;
}
std::vector<EdgeWrapper *> findEndEdges(
    std::vector<EdgeWrapper *> &edge_repository) {
  std::vector<EdgeWrapper *> end_edge;
  for (auto edge_wrapper : edge_repository) {
    if (edge_wrapper->consumers_.empty()) {
      end_edge.emplace_back(edge_wrapper);
    }
  }
  return end_edge;
}

Node *getNode(std::vector<NodeWrapper *> &node_repository,
              const std::string &node_name) {
  for (auto node_wrapper : node_repository) {
    if (node_wrapper->name_ == node_name) {
      return node_wrapper->node_;
    }
  }
  return nullptr;
}
NodeWrapper *findNodeWrapper(std::vector<NodeWrapper *> &node_repository,
                             const std::string &node_name) {
  for (auto node_wrapper : node_repository) {
    if (node_wrapper->name_ == node_name) {
      return node_wrapper;
    }
  }
  return nullptr;
}
NodeWrapper *findNodeWrapper(std::vector<NodeWrapper *> &node_repository,
                             Node *node) {
  for (auto node_wrapper : node_repository) {
    if (node_wrapper->node_ == node) {
      return node_wrapper;
    }
  }
  return nullptr;
}
std::vector<NodeWrapper *> findStartNodes(
    std::vector<NodeWrapper *> &node_repository) {
  std::vector<NodeWrapper *> start_nodes;
  for (auto node_wrapper : node_repository) {
    if (node_wrapper->predecessors_.empty()) {
      start_nodes.emplace_back(node_wrapper);
    }
  }
  return start_nodes;
}
// 这个实现是不充分的， 有些边可以既是输出边也是中间节点的输入边
std::vector<NodeWrapper *> findEndNodes(
    std::vector<NodeWrapper *> &node_repository) {
  std::vector<NodeWrapper *> end_nodes;
  for (auto node_wrapper : node_repository) {
    if (node_wrapper->successors_.empty()) {
      end_nodes.emplace_back(node_wrapper);
    }
  }
  return end_nodes;
}

base::Status setColor(std::vector<NodeWrapper *> &node_repository,
                      base::NodeColorType color) {
  for (auto node_wrapper : node_repository) {
    node_wrapper->color_ = color;
  }
  return base::kStatusCodeOk;
}

base::Status dumpDag(std::vector<EdgeWrapper *> &edge_repository,
                     std::vector<NodeWrapper *> &node_repository,
                     std::vector<Edge *> &graph_inputs,
                     std::vector<Edge *> &graph_outputs,
                     const std::string &name, std::ostream &oss) {
  base::Status status = base::kStatusCodeOk;
  // NNDEPLOY_LOGI("#######################\n");
  // NNDEPLOY_LOGI("Node dump Phase!\n");
  // NNDEPLOY_LOGI("#######################\n");
  if (name.empty()) {
    oss << "digraph graph {\n";
  } else {
    std::string label = "\"" + name + "\"";
    oss << "digraph " << label << " {\n";
  }
  for (auto input : graph_inputs) {
    if (input->getName().empty()) {
      oss << "p" << (void *)input << "[shape=box, label=input]\n";
    } else {
      std::string label = "\"" + input->getName() + "\"";
      oss << "p" << (void *)input << "[shape=box, label=" << label << "]\n";
    }
    EdgeWrapper *edge_wrapper = findEdgeWrapper(edge_repository, input);
    for (auto node_wrapper : edge_wrapper->consumers_) {
      auto node = node_wrapper->node_;
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
  for (auto node_wrapper : node_repository) {
    Node *node = node_wrapper->node_;
    if (node->getName().empty()) {
      oss << "p" << (void *)node << "[label=node]\n";
    } else {
      std::string label = "\"" + node->getName() + "\"";
      oss << "p" << (void *)node << "[label=" << label << "]\n";
    }
    for (auto successor : node_wrapper->successors_) {
      auto outputs = node->getAllOutput();
      auto inputs = successor->node_->getAllInput();
      // 两Node间可能有多条Edge
      for (auto output : outputs) {
        Edge *out_in = nullptr;
        for (auto input : inputs) {
          if (output == input) {
            out_in = output;
          }
        }
        if (out_in != nullptr) {
          oss << "p" << (void *)node << "->"
              << "p" << (void *)(successor->node_);
          if (out_in->getName().empty()) {
            oss << "\n";
          } else {
            std::string label = "\"" + out_in->getName() + "\"";
            oss << "[label=" << label << "]\n";
          }
        }
      }
    }
  }
  for (auto output : graph_outputs) {
    if (output->getName().empty()) {
      oss << "p" << (void *)output << "[shape=box, label=output]\n";
    } else {
      std::string label = "\"" + output->getName() + "\"";
      oss << "p" << (void *)output << "[shape=box, label=" << label << "]\n";
    }
    EdgeWrapper *edge_wrapper = findEdgeWrapper(edge_repository, output);
    for (auto node_wrapper : edge_wrapper->producers_) {
      auto node = node_wrapper->node_;
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
  oss << "}\n";
  return status;
}

std::vector<NodeWrapper *> checkUnuseNode(
    std::vector<NodeWrapper *> &node_repository) {
  std::vector<NodeWrapper *> unused;
  for (auto node_wrapper : node_repository) {
    if (node_wrapper->color_ == base::kNodeColorWhite) {
      NNDEPLOY_LOGE("Unuse node found in graph, Node name: %s.",
                    node_wrapper->name_.c_str());
      unused.emplace_back(node_wrapper);
    }
  }
  return unused;
}
std::vector<EdgeWrapper *> checkUnuseEdge(
    std::vector<NodeWrapper *> &node_repository,
    std::vector<EdgeWrapper *> &edge_repository) {
  std::vector<EdgeWrapper *> unused_edges;
  std::vector<NodeWrapper *> unused_nodes = checkUnuseNode(node_repository);
  for (auto iter : unused_nodes) {
    for (auto iter_input : iter->node_->getAllInput()) {
      EdgeWrapper *input_wrapper = findEdgeWrapper(edge_repository, iter_input);
      if (input_wrapper != nullptr) {
        NNDEPLOY_LOGE("Unuse edge found in graph, edge name: %s.",
                      input_wrapper->name_.c_str());
        unused_edges.emplace_back(input_wrapper);
      }
    }
    for (auto iter_output : iter->node_->getAllOutput()) {
      EdgeWrapper *output_wrapper =
          findEdgeWrapper(edge_repository, iter_output);
      if (output_wrapper != nullptr) {
        NNDEPLOY_LOGE("Unuse edge found in graph, edge name: %s.",
                      output_wrapper->name_.c_str());
        unused_edges.emplace_back(output_wrapper);
      }
    }
  }
  for (auto edge_wrapper : edge_repository) {
    if (edge_wrapper->producers_.empty() && edge_wrapper->consumers_.empty()) {
      NNDEPLOY_LOGE("Unuse edge found in graph, edge name: %s.",
                    edge_wrapper->name_.c_str());
      unused_edges.emplace_back(edge_wrapper);
    }
  }
  return unused_edges;
}

base::Status topoSortBFS(std::vector<NodeWrapper *> &node_repository,
                         std::vector<NodeWrapper *> &topo_sort_node) {
  std::vector<NodeWrapper *> start_nodes = findStartNodes(node_repository);
  if (start_nodes.empty()) {
    NNDEPLOY_LOGE("No start node found in graph");
    return base::kStatusCodeErrorInvalidValue;
  }
  std::deque<NodeWrapper *> node_deque;
  for (auto node_wrapper : start_nodes) {
    node_wrapper->color_ = base::kNodeColorGray;
    node_deque.emplace_back(node_wrapper);
  }
  while (!node_deque.empty()) {
    NodeWrapper *node_wrapper = node_deque.front();
    for (auto successor : node_wrapper->successors_) {
      if (successor->color_ == base::kNodeColorWhite) {
        successor->color_ = base::kNodeColorGray;
        node_deque.emplace_back(successor);
      } else if (successor->color_ == base::kNodeColorGray) {
        continue;
      } else {
        NNDEPLOY_LOGE("Cycle detected in graph");
        return base::kStatusCodeErrorInvalidValue;
      }
    }
    node_deque.pop_front();
    node_wrapper->color_ = base::kNodeColorBlack;
    topo_sort_node.emplace_back(node_wrapper);
  }

  checkUnuseNode(node_repository);

  return base::kStatusCodeOk;
}

base::Status TopoSortDFSRecursive(NodeWrapper *node_wrapper,
                                  std::stack<NodeWrapper *> &dst) {
  base::Status status = base::kStatusCodeOk;
  node_wrapper->color_ = base::kNodeColorGray;
  for (auto successor : node_wrapper->successors_) {
    if (successor->color_ == base::kNodeColorWhite) {
      status = TopoSortDFSRecursive(successor, dst);
      NNDEPLOY_RETURN_ON_NEQ(status, base::kStatusCodeOk,
                             "Cycle detected in graph");
    } else if (successor->color_ == base::kNodeColorGray) {
      NNDEPLOY_LOGE("Cycle detected in graph");
      return base::kStatusCodeErrorInvalidValue;
    } else {
      continue;
    }
  }
  node_wrapper->color_ = base::kNodeColorBlack;
  dst.push(node_wrapper);
  return status;
}

base::Status topoSortDFS(std::vector<NodeWrapper *> &node_repository,
                         std::vector<NodeWrapper *> &topo_sort_node) {
  base::Status status = base::kStatusCodeOk;
  std::vector<NodeWrapper *> start_nodes = findStartNodes(node_repository);
  if (start_nodes.empty()) {
    NNDEPLOY_LOGE("No start node found in graph");
    return base::kStatusCodeErrorInvalidValue;
  }
  std::stack<NodeWrapper *> dst;
  for (auto node_wrapper : start_nodes) {
    if (node_wrapper->color_ == base::kNodeColorWhite) {
      status = TopoSortDFSRecursive(node_wrapper, dst);
    } else if (node_wrapper->color_ == base::kNodeColorGray) {
      NNDEPLOY_LOGE("Cycle detected in graph");
      status = base::kStatusCodeErrorInvalidValue;
    } else {
      continue;
    }
  }
  while (!dst.empty()) {
    topo_sort_node.emplace_back(dst.top());
    dst.pop();
  }

  checkUnuseNode(node_repository);

  return base::kStatusCodeOk;
}

base::Status topoSort(std::vector<NodeWrapper *> &node_repository,
                      base::TopoSortType topo_sort_type,
                      std::vector<NodeWrapper *> &topo_sort_node) {
  base::Status status = base::kStatusCodeOk;
  if (topo_sort_type == base::kTopoSortTypeBFS) {
    status = topoSortBFS(node_repository, topo_sort_node);
    if (status != base::kStatusCodeOk) {
      return status;
    }
  } else if (topo_sort_type == base::kTopoSortTypeDFS) {
    status = topoSortDFS(node_repository, topo_sort_node);
    if (status != base::kStatusCodeOk) {
      return status;
    }
  } else {
    NNDEPLOY_LOGE("Invalid topo sort type");
    return base::kStatusCodeErrorInvalidValue;
  }
  return status;
}

bool checkEdge(const std::vector<Edge *> &src_edges,
               const std::vector<Edge *> &dst_edges) {
  for (auto edge : src_edges) {
    bool flag = false;
    for (auto check_edge : dst_edges) {
      if (edge == check_edge) {
        flag = true;
        break;
      }
    }
    if (!flag) {
      return false;
    }
  }
  return true;
}

}  // namespace dag
}  // namespace nndeploy