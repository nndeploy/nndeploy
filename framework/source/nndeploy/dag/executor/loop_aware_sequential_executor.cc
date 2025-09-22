#include "nndeploy/dag/executor/loop_aware_sequential_executor.h"

namespace nndeploy {
namespace dag {

LoopAwareSequentialExecutor::LoopAwareSequentialExecutor() : Executor() {};
LoopAwareSequentialExecutor::~LoopAwareSequentialExecutor() {};

bool LoopAwareSequentialExecutor::buildTopoIgnoringFeedback_(
  const std::vector<EdgeWrapper*>& edges,
  const std::vector<NodeWrapper*>& nodes,
  std::vector<NodeWrapper*>& topo_out) {

topo_out.clear();
if (nodes.empty()) return true;

// 1) 建索引
std::unordered_map<NodeWrapper*, int> idx;
idx.reserve(nodes.size());
for (int i = 0; i < (int)nodes.size(); ++i) idx[nodes[i]] = i;

// 2) 构建“忽略 feedback”的邻接与入度
std::vector<int> indeg(nodes.size(), 0);
std::vector<std::vector<int>> adj(nodes.size());
for (auto* ew : edges) {
  if (!ew || !ew->edge_) continue;
  if (ew->edge_->isFeedback()) continue;  // 忽略 feedback 边
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
  int u = q.front(); q.pop();
  topo_out.push_back(nodes[u]);
  for (int v : adj[u]) {
    if (--indeg[v] == 0) q.push(v);
  }
}

// 4) 如果因为用户没标干净 feedback 导致仍有环，兜底把剩下的也 append 进来
if ((int)topo_out.size() != (int)nodes.size()) {
  NNDEPLOY_LOGW(
      "LoopAwareSequentialExecutor: non-feedback subgraph still cyclic; "
      "falling back by appending remaining nodes.\n");
  std::vector<char> used(nodes.size(), 0);
  for (auto* nw : topo_out) used[idx[nw]] = 1;
  for (int i = 0; i < (int)nodes.size(); ++i) {
    if (!used[i]) topo_out.push_back(nodes[i]);
  }
}
return true;
}

base::Status LoopAwareSequentialExecutor::init(
  std::vector<EdgeWrapper*>& edge_repository,
  std::vector<NodeWrapper*>& node_repository) {
base::Status status = base::kStatusCodeOk;

// 先尝试基于“忽略 feedback”的拓扑
if (!buildTopoIgnoringFeedback_(edge_repository, node_repository,
                                topo_sort_node_)) {
  // 理论上不会走到这里；兜底仍可用 DFS
  status = topoSortDFS(node_repository, topo_sort_node_);
  if (status != base::kStatusCodeOk) {
    NNDEPLOY_LOGE("LoopAwareSequentialExecutor: topo sort failed.\n");
    return status;
  }
}

// 节点 init（按 topo）
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

base::Status LoopAwareSequentialExecutor::deinit() {
base::Status status = base::kStatusCodeOk;

// 请求边终止（与原实现一致）
for (auto* ew : edge_repository_) {
  if (!ew || !ew->edge_) continue;
  if (!ew->edge_->requestTerminate()) {
    NNDEPLOY_LOGE("LoopAwareSequentialExecutor: requestTerminate() failed!\n");
    return base::kStatusCodeErrorDag;
  }
}

// 反向析构节点
for (auto* nw : topo_sort_node_) {
  Node* node = nw->node_;
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

base::Status LoopAwareSequentialExecutor::sweepOnce_(bool& progressed) {
progressed = false;

for (auto* nw : topo_sort_node_) {
  Node* node = nw->node_;
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
    // 本轮无需跑；继续下一个
    continue;
  } else {
    // 不是 Complete 也不是 Terminate，说明输入暂未就绪；继续下一个
    // 这里不直接报错，给下一轮机会（这正是“逐轮推进”的关键）
    continue;
  }
}

return base::kStatusCodeOk;
}

base::Status LoopAwareSequentialExecutor::run() {
base::Status status = base::kStatusCodeOk;

int rounds = 0;
for (;;) {
  bool progressed = false;
  status = sweepOnce_(progressed);
  if (status != base::kStatusCodeOk) return status;

  if (!progressed) {
    // 整轮没有任何节点运行 => 没有新的可推进工作，退出
    break;
  }
  rounds++;

  if (max_rounds_ > 0 && rounds >= max_rounds_) {
    NNDEPLOY_LOGW("LoopAwareSequentialExecutor: reach max_rounds_=%d, break.\n", max_rounds_);
    break;
  }
}
return status;
}

bool LoopAwareSequentialExecutor::synchronize() {
for (auto* nw : topo_sort_node_) {
  Node* node = nw->node_;
  if (!node) continue;
  if (!node->synchronize()) return false;
}
return true;
}

bool LoopAwareSequentialExecutor::interrupt() {
for (auto* nw : topo_sort_node_) {
  Node* node = nw->node_;
  if (!node) continue;
  if (!node->interrupt()) return false;
}
return true;
}

}  // namespace dag
}  // namespace nndeploy