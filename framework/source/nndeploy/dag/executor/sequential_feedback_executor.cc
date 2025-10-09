#include "nndeploy/dag/executor/sequential_feedback_executor.h"

namespace nndeploy {
namespace dag {

SequentialFeedbackExecutor::SequentialFeedbackExecutor() : Executor() {};
SequentialFeedbackExecutor::~SequentialFeedbackExecutor() {};

bool SequentialFeedbackExecutor::buildTopoIgnoringFeedback_(
    const std::vector<EdgeWrapper*>& edges,
    const std::vector<NodeWrapper*>& nodes,
    std::vector<NodeWrapper*>& topo_out) {
  topo_out.clear();
  if (nodes.empty()) return true;

  std::unordered_map<NodeWrapper*, int> idx;
  idx.reserve(nodes.size());
  for (int i = 0; i < (int)nodes.size(); ++i) idx[nodes[i]] = i;

  std::vector<int> indeg(nodes.size(), 0);
  std::vector<std::vector<int>> adj(nodes.size());
  for (auto* ew : edges) {
    if (!ew || !ew->edge_) continue;
    if (ew->edge_->isFeedback()) continue;
    if (ew->producers_.empty() || ew->consumers_.empty()) continue;

    for (auto* p : ew->producers_) {
      auto itp = idx.find(p);
      if (itp == idx.end()) continue;
      int u = itp->second;
      for (auto* c : ew->consumers_) {
        auto itc = idx.find(c);
        if (itc == idx.end()) continue;
        int v = itc->second;
        adj[u].push_back(v);
        indeg[v]++;
      }
    }
  }

  // 3) Kahn
  std::queue<int> q;
  for (int i = 0; i < (int)nodes.size(); ++i) {
    if (indeg[i] == 0) q.push(i);
  }
  while (!q.empty()) {
    int u = q.front();
    q.pop();
    topo_out.push_back(nodes[u]);
    for (int v : adj[u]) {
      if (--indeg[v] == 0) q.push(v);
    }
  }

  if ((int)topo_out.size() != (int)nodes.size()) {
    NNDEPLOY_LOGW(
        "SequentialFeedbackExecutor: non-feedback subgraph still cyclic; "
        "falling back by appending remaining nodes.\n");
    std::vector<char> used(nodes.size(), 0);
    for (auto* nw : topo_out) used[idx[nw]] = 1;
    for (int i = 0; i < (int)nodes.size(); ++i) {
      if (!used[i]) topo_out.push_back(nodes[i]);
    }
  }
  return true;
}

base::Status SequentialFeedbackExecutor::init(
    std::vector<EdgeWrapper*>& edge_repository,
    std::vector<NodeWrapper*>& node_repository) {
  base::Status status = base::kStatusCodeOk;

  if (!buildTopoIgnoringFeedback_(edge_repository, node_repository,
                                  topo_sort_node_)) {
    status = topoSortDFS(node_repository, topo_sort_node_);
    if (status != base::kStatusCodeOk) {
      NNDEPLOY_LOGE("SequentialFeedbackExecutor: topo sort failed.\n");
      return status;
    }
  }

  for (auto* nw : topo_sort_node_) {
    Node* node = nw->node_;
    if (!node) continue;
    if (node->getInitialized()) continue;

    if (node->checkInterruptStatus() == true) {
      node->setRunningFlag(false);
      return base::kStatusCodeNodeInterrupt;
    }
    node->setInitializedFlag(false);
    status = node->init();
    if (status != base::kStatusCodeOk) {
      NNDEPLOY_LOGE("Node %s init failed\n", node->getName().c_str());
      return status;
    }
    node->setInitializedFlag(true);
  }

  edge_repository_ = edge_repository;
  return status;
}

base::Status SequentialFeedbackExecutor::deinit() {
  base::Status status = base::kStatusCodeOk;

  for (auto* e : edge_repository_) {
    if (!e || !e->edge_) continue;
    if (!e->edge_->requestTerminate()) {
      NNDEPLOY_LOGE("SequentialFeedbackExecutor: requestTerminate() failed!\n");
      return base::kStatusCodeErrorDag;
    }
  }

  for (auto* n : topo_sort_node_) {
    Node* node = n->node_;
    if (!node || !node->getInitialized()) continue;
    status = node->deinit();
    if (status != base::kStatusCodeOk) {
      NNDEPLOY_LOGE("node[%s] deinit failed\n", node->getName().c_str());
      return status;
    }
    node->setInitializedFlag(false);
  }
  return status;
}

base::Status SequentialFeedbackExecutor::run_once(bool& progressed) {
  progressed = false;

  for (auto* n : topo_sort_node_) {
    Node* node = n->node_;
    if (!node) continue;

    if (node->checkInterruptStatus() == true) {
      node->setRunningFlag(false);
      return base::kStatusCodeNodeInterrupt;
    }

    base::EdgeUpdateFlag flag = node->updateInput();
    if (flag == base::kEdgeUpdateFlagComplete) {
      node->setRunningFlag(true);
      base::Status st = node->run();
      node->setRunningFlag(false);
      if (st != base::kStatusCodeOk) {
        NNDEPLOY_LOGE("node[%s] run failed\n", node->getName().c_str());
        return st;
      }
      progressed = true;
    } else if (flag == base::kEdgeUpdateFlagTerminate) {
      continue;
    } else {
      continue;
    }
  }

  return base::kStatusCodeOk;
}

base::Status SequentialFeedbackExecutor::run() {
  base::Status status = base::kStatusCodeOk;
  bool progressed = false;
  status = run_once(progressed);
  if (status != base::kStatusCodeOk) return status;

  // dataflow, running according to the input edge
  // int rounds = 0;
  // for (;;) {
  //   // NNDEPLOY_LOGE("start run\n");
  //   bool progressed = false;
  //   status = run_once(progressed);
  //   if (status != base::kStatusCodeOk) return status;

  //   if (!progressed) {
  //     // 整轮没有任何节点运行 => 没有新的可推进工作，退出
  //     break;
  //   }
  //   rounds++;

  //   if (max_rounds_ > 0 && rounds >= max_rounds_) {
  //     NNDEPLOY_LOGW(
  //         "SequentialFeedbackExecutor: reach max_rounds_=%d, break.\n",
  //         max_rounds_);
  //     break;
  //   }
  // }
  return status;
}

bool SequentialFeedbackExecutor::synchronize() {
  for (auto* nw : topo_sort_node_) {
    Node* node = nw->node_;
    if (!node) continue;
    if (!node->synchronize()) return false;
  }
  return true;
}

bool SequentialFeedbackExecutor::interrupt() {
  for (auto* nw : topo_sort_node_) {
    Node* node = nw->node_;
    if (!node) continue;
    if (!node->interrupt()) return false;
  }
  return true;
}

}  // namespace dag
}  // namespace nndeploy