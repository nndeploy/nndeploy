#ifndef AF95ABAC_870C_46DF_9DFA_C4C202F6EF43
#define AF95ABAC_870C_46DF_9DFA_C4C202F6EF43

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

#endif /* AF95ABAC_870C_46DF_9DFA_C4C202F6EF43 */
