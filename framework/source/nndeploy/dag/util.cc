
#include "nndeploy/dag/util.h"

#include "nndeploy/dag/graph.h"

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
    oss << "digraph " << name << " {\n";
  }
  for (auto input : graph_inputs) {
    if (input->getName().empty()) {
      oss << "p" << (void *)input << "[shape=box, label=input]\n";
    } else {
      oss << "p" << (void *)input << "[shape=box, label=" << input->getName()
          << "]\n";
    }
    EdgeWrapper *edge_wrapper = findEdgeWrapper(edge_repository, input);
    for (auto node_wrapper : edge_wrapper->consumers_) {
      auto node = node_wrapper->node_;
      oss << "p" << (void *)input << "->"
          << "p" << (void *)node;
      if (input->getName().empty()) {
        oss << "\n";
      } else {
        oss << "[label=" << input->getName() << "]\n";
      }
    }
  }
  for (auto node_wrapper : node_repository) {
    Node *node = node_wrapper->node_;
    if (node->getName().empty()) {
      oss << "p" << (void *)node << "[label=node]\n";
    } else {
      oss << "p" << (void *)node << "[label=" << node->getName() << "]\n";
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
            oss << "[label=" << out_in->getName() << "]\n";
          }
        }
      }
    }
  }
  for (auto output : graph_outputs) {
    if (output->getName().empty()) {
      oss << "p" << (void *)output << "[shape=box, label=output]\n";
    } else {
      oss << "p" << (void *)output << "[shape=box, label=" << output->getName()
          << "]\n";
    }
    EdgeWrapper *edge_wrapper = findEdgeWrapper(edge_repository, output);
    for (auto node_wrapper : edge_wrapper->producers_) {
      auto node = node_wrapper->node_;
      oss << "p" << (void *)node << "->"
          << "p" << (void *)output;
      if (output->getName().empty()) {
        oss << "\n";
      } else {
        oss << "[label=" << output->getName() << "]\n";
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
      NNDEPLOY_LOGE("Unuse node found in graph, Node name: %s.\n",
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
  // 找到所有入度为0的节点作为起始节点
  std::vector<NodeWrapper *> start_nodes = findStartNodes(node_repository);
  if (start_nodes.empty()) {
    NNDEPLOY_LOGE("No start node found in graph!\n");
    return base::kStatusCodeErrorInvalidValue;
  }

  // 记录每个节点的入度
  std::unordered_map<NodeWrapper *, int> in_degree;
  for (auto node : node_repository) {
    in_degree[node] = node->predecessors_.size();
  }

  // 将所有入度为0的节点加入队列
  std::queue<NodeWrapper *> q;
  for (auto node : start_nodes) {
    q.push(node);
  }

  // BFS遍历
  while (!q.empty()) {
    NodeWrapper *cur = q.front();
    cur->color_ = base::kNodeColorBlack;
    q.pop();
    topo_sort_node.push_back(cur);

    // 将当前节点的所有后继节点的入度减1
    for (auto succ : cur->successors_) {
      in_degree[succ]--;
      // 如果入度变为0,加入队列
      if (in_degree[succ] == 0) {
        q.push(succ);
      }
    }
  }

  // 检查是否存在环
  if (topo_sort_node.size() != node_repository.size()) {
    NNDEPLOY_LOGE("Cycle detected in graph!\n");
    return base::kStatusCodeErrorInvalidValue;
  }

  // 检查未使用的节点
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

void findConsumerNode(EdgeWrapper *edge_wrapper,
                      std::vector<Node *> &consumers) {
  for (auto consumer : edge_wrapper->consumers_) {
    auto consumer_node = consumer->node_;
    if (consumer_node->getGraphFlag()) {
      Graph *graph = (Graph *)consumer_node;
      EdgeWrapper *inner_edge_wrapper =
          graph->getEdgeWrapper(edge_wrapper->edge_);
      if (inner_edge_wrapper == nullptr) {
        continue;
      }
      // 递归查找子图中的消费者节点
      std::vector<Node *> inner_consumers;
      findConsumerNode(inner_edge_wrapper, inner_consumers);
      // 将子图中找到的所有消费者节点添加到结果中
      consumers.insert(consumers.end(), inner_consumers.begin(),
                       inner_consumers.end());
    } else {
      consumers.emplace_back(consumer_node);
    }
  }
  return;
}

void findProducerNode(EdgeWrapper *edge_wrapper,
                      std::vector<Node *> &producers) {
  for (auto producer : edge_wrapper->producers_) {
    auto producer_node = producer->node_;
    if (producer_node->getGraphFlag()) {
      Graph *graph = (Graph *)producer_node;
      EdgeWrapper *inner_edge_wrapper =
          graph->getEdgeWrapper(edge_wrapper->edge_);
      if (inner_edge_wrapper == nullptr) {
        continue;
      }
      // 递归查找子图中的生产者节点
      std::vector<Node *> inner_producers;
      findProducerNode(inner_edge_wrapper, inner_producers);
      // 将子图中找到的所有生产者节点添加到结果中
      producers.insert(producers.end(), inner_producers.begin(),
                       inner_producers.end());
    } else {
      producers.emplace_back(producer_node);
    }
  }
  return;
}
// 修改函数签名，添加 allocator 参数
void replaceJsonValue(rapidjson::Value &json, const std::string &key,
                      const std::string &new_value,
                      rapidjson::Document::AllocatorType &allocator) {
  if (!json.IsObject()) return;

  if (json.HasMember(key.c_str())) {
    NNDEPLOY_LOGI("replaceJsonValue: %s, %s\n", key.c_str(), new_value.c_str());
    if (json[key.c_str()].IsString()) {
      json[key.c_str()].SetString(new_value.c_str(), new_value.length(),
                                  allocator);
    } else if (json[key.c_str()].IsInt()) {
      int int_val = std::stoi(new_value);
      json[key.c_str()].SetInt(int_val);
    } else if (json[key.c_str()].IsInt64()) {
      int64_t int64_val = std::stoll(new_value);
      json[key.c_str()].SetInt64(int64_val);
    } else if (json[key.c_str()].IsUint()) {
      unsigned int uint_val = std::stoul(new_value);
      json[key.c_str()].SetUint(uint_val);
    } else if (json[key.c_str()].IsUint64()) {
      uint64_t uint64_val = std::stoull(new_value);
      json[key.c_str()].SetUint64(uint64_val);
    } else if (json[key.c_str()].IsFloat()) {
      float float_val = std::stof(new_value);
      json[key.c_str()].SetFloat(float_val);
    } else if (json[key.c_str()].IsDouble()) {
      double double_val = std::stod(new_value);
      json[key.c_str()].SetDouble(double_val);
    } else if (json[key.c_str()].IsBool()) {
      bool bool_val = (new_value == "true" || new_value == "1");
      json[key.c_str()].SetBool(bool_val);
    } else if (json[key.c_str()].IsObject()) {
      // 对象类型：解析new_value为JSON对象
      rapidjson::Document doc;
      doc.Parse(new_value.c_str());
      if (!doc.HasParseError() && doc.IsObject()) {
        json[key.c_str()].CopyFrom(doc, allocator);
      }
    } else if (json[key.c_str()].IsArray()) {
      // 数组类型：解析new_value为JSON数组
      rapidjson::Document doc;
      doc.Parse(new_value.c_str());
      if (!doc.HasParseError() && doc.IsArray()) {
        json[key.c_str()].CopyFrom(doc, allocator);
      }
    } else {
      // 其他类型：尝试解析为JSON并替换
      rapidjson::Document doc;
      doc.Parse(new_value.c_str());
      if (!doc.HasParseError()) {
        json[key.c_str()].CopyFrom(doc, allocator);
      }
    }
    return;
  }

  if (json.HasMember("param_") && json["param_"].IsObject()) {
    NNDEPLOY_LOGI("replaceGraphJsonObj: %s, %s\n", key.c_str(), new_value.c_str());
    replaceJsonValue(json["param_"], key, new_value, allocator);
    return;
  }
}

std::string replaceGraphJsonStr(
    std::map<std::string, std::map<std::string, std::string>> node_value_map,
    const std::string &json_str) {
  rapidjson::Document document;
  if (document.Parse(json_str.c_str()).HasParseError()) {
    NNDEPLOY_LOGE("parse json string failed\n");
    return json_str;
  }
  rapidjson::Value &json = document;
  replaceGraphJsonObj(node_value_map, json, document.GetAllocator());

  // 非美化
  rapidjson::StringBuffer buffer;
  rapidjson::Writer<rapidjson::StringBuffer> writer(buffer);
  json.Accept(writer);
  std::string json_result = buffer.GetString();
  
  // 美化json_result
  // rapidjson::StringBuffer buffer;
  // rapidjson::PrettyWriter<rapidjson::StringBuffer> writer(buffer);
  // json.Accept(writer);
  // std::string json_result = buffer.GetString();
  
  // NNDEPLOY_LOGI("replaceGraphJsonStr: %s\n", json_result.c_str());
  return json_result;
}

void replaceGraphJsonObj(
    std::map<std::string, std::map<std::string, std::string>> node_value_map,
    rapidjson::Value &json,
    rapidjson::Document::AllocatorType &allocator) {
  if (json.IsObject()) {
    if (json.HasMember("name_") && json["name_"].IsString()) {
      std::string name = json["name_"].GetString();
      auto node_iter = node_value_map.find(name);
      if (node_iter != node_value_map.end()) {
        for (const auto &param_pair : node_iter->second) {
          const std::string &key = param_pair.first;
          const std::string &new_value = param_pair.second;
          NNDEPLOY_LOGI("replaceGraphJsonObj: %s, %s, %s\n", name.c_str(), key.c_str(), new_value.c_str());
          replaceJsonValue(json, key, new_value, allocator);
        }
      }
    }

    if (json.HasMember("node_repository_") &&
        json["node_repository_"].IsArray()) {
      for (auto &node : json["node_repository_"].GetArray()) {
        replaceGraphJsonObj(node_value_map, node, allocator);
      }
    }
  }
  return;
}

}  // namespace dag
}  // namespace nndeploy