#include "nndeploy_api_registry.h"

namespace nndeploy {
namespace {
using SubModuleMap =
    std::map<std::string, std::vector<std::function<void(pybind11::module&)>>>;
SubModuleMap* GetSubModuleMap() {
  static SubModuleMap sub_module_map;
  return &sub_module_map;
}
}  // namespace

void NndeployModuleRegistry::Register(
    std::string module_path,
    std::function<void(pybind11::module&)> BuildModule) {
  (*GetSubModuleMap())[module_path].emplace_back(BuildModule);
}
void NndeployModuleRegistry::ImportAll(pybind11::module& m) {
  // 预定义模块加载顺序,使用unordered_set提高查找效率
  const std::vector<std::string> module_order = {
    "base",
    "thread_pool", 
    "device",
    "ir",
    "op",
    "net", 
    "inference",
    "dag",
    "codec",
    "preprocess",
    "tokenizer",
    "infer",
    "classifier", 
    "detect",
    "segment",
    "track",
    "ocr",
    "llm",
    "stable_diffusion",
  };

  std::unordered_set<std::string> ordered_modules(module_order.begin(), module_order.end());
  auto* sub_module_map = GetSubModuleMap();

  // 按预定义顺序加载模块
  for (const auto& module_name : module_order) {
    auto it = sub_module_map->find(module_name);
    if (it != sub_module_map->end()) {
      for (const auto& BuildModule : it->second) {
        BuildSubModule(it->first, m, BuildModule);
      }
    }
  }

  // 加载其他未在预定义顺序中的模块
  for (const auto& pair : *sub_module_map) {
    if (!ordered_modules.count(pair.first)) {
      for (const auto& BuildModule : pair.second) {
        BuildSubModule(pair.first, m, BuildModule);
      }
    }
  }
}

void NndeployModuleRegistry::BuildSubModule(
    const std::string& module_path, pybind11::module& m,
    const std::function<void(pybind11::module&)>& BuildModule) {
  if (module_path.empty()) {
    BuildModule(m);
    return;
  }
  size_t dot_pos = module_path.find(".");
  if (dot_pos == std::string::npos) {
    pybind11::module sub_module = m.def_submodule(module_path.data());
    BuildModule(sub_module);
  } else {
    const std::string& sub_module_name = module_path.substr(0, dot_pos);
    pybind11::module sub_module = m.def_submodule(sub_module_name.data());
    BuildSubModule(module_path.substr(dot_pos + 1), sub_module, BuildModule);
  }
}

}  // namespace nndeploy