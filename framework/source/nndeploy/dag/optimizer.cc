
#include "nndeploy/dag/optimizer.h"

#include <algorithm>
#include <map>
#include <set>

#include "nndeploy/base/log.h"
#include "nndeploy/dag/graph.h"

namespace nndeploy {
namespace dag {

// ---------------------------------------------------------------------------
// Global registry
// ---------------------------------------------------------------------------

static std::map<std::string, DagOptPassCreator>& GetPassRegistry() {
  static std::map<std::string, DagOptPassCreator> registry;
  return registry;
}

void registerDagOptPass(const std::string& name, DagOptPassCreator creator) {
  GetPassRegistry()[name] = std::move(creator);
}

std::shared_ptr<DagOptPass> createDagOptPass(const std::string& name) {
  auto& registry = GetPassRegistry();
  auto it = registry.find(name);
  if (it != registry.end()) {
    return it->second();
  }
  return nullptr;
}

std::vector<std::string> getAllRegisteredDagOptPasses() {
  auto& registry = GetPassRegistry();
  std::vector<std::string> names;
  names.reserve(registry.size());
  for (auto& kv : registry) {
    names.push_back(kv.first);
  }
  return names;
}

// ---------------------------------------------------------------------------
// DagOptimizer
// ---------------------------------------------------------------------------

base::Status DagOptimizer::init(
    const std::vector<std::string>& enabled_passes,
    const std::vector<std::string>& disabled_passes) {
  // 三态逻辑（对齐 Net::optimizer 的 enable/disable 设计）：
  //   模式A（白名单）：enabled_passes 非空 → 只运行指定的 pass
  //   模式B（黑名单）：enabled_passes 为空 + disabled_passes 非空 → 运行所有，排除禁用的
  //   模式C（全运行）：都为空 → 运行所有已注册的 pass

  std::set<std::string> disabled_set(disabled_passes.begin(),
                                     disabled_passes.end());

  std::vector<std::string> passes_to_run;
  if (!enabled_passes.empty()) {
    // 模式A：白名单
    passes_to_run = enabled_passes;
  } else {
    // 模式B 或 模式C：从注册表收集所有 pass
    passes_to_run = getAllRegisteredDagOptPasses();
    if (!disabled_set.empty()) {
      // 模式B：移除禁用的
      auto it = std::remove_if(passes_to_run.begin(), passes_to_run.end(),
                               [&disabled_set](const std::string& name) {
                                 return disabled_set.count(name) > 0;
                               });
      passes_to_run.erase(it, passes_to_run.end());
    }
  }

  for (const auto& pass_name : passes_to_run) {
    auto pass = createDagOptPass(pass_name);
    if (pass != nullptr) {
      passes_.push_back(std::move(pass));
      NNDEPLOY_LOGI("DagOptimizer enabled pass [%s]\n", pass_name.c_str());
    } else {
      NNDEPLOY_LOGW("DagOptimizer pass [%s] not registered, skipped\n",
                     pass_name.c_str());
    }
  }
  return base::kStatusCodeOk;
}

base::Status DagOptimizer::optimize(Graph* graph) {
  if (graph == nullptr) {
    NNDEPLOY_LOGE("DagOptimizer::optimize got null graph\n");
    return base::kStatusCodeErrorInvalidValue;
  }
  for (auto& pass : passes_) {
    NNDEPLOY_LOGI("DagOptimizer running pass [%s]\n", pass->getName().c_str());
    base::Status status = pass->optimize(graph);
    if (status != base::kStatusCodeOk) {
      NNDEPLOY_LOGE("DagOptimizer pass [%s] failed\n", pass->getName().c_str());
      return status;
    }
  }
  return base::kStatusCodeOk;
}

base::Status DagOptimizer::deinit() {
  passes_.clear();
  return base::kStatusCodeOk;
}

}  // namespace dag
}  // namespace nndeploy
