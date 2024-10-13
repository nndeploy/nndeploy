#ifndef _NNDEPLOY_PYTHON_SRC_NNDEPLOY_API_REGISTRY_h_
#define _NNDEPLOY_PYTHON_SRC_NNDEPLOY_API_REGISTRY_h_

#include <pybind11/pybind11.h>

#include <functional>
#include <map>
#include <vector>

namespace py = pybind11;

namespace nndeploy {

class NndeployModuleRegistry {
 public:
  NndeployModuleRegistry() = default;
  ~NndeployModuleRegistry() = default;

  void Register(std::string module_path,
                std::function<void(pybind11::module&)> BuildModule);
  void ImportAll(pybind11::module& m);

 private:
  void BuildSubModule(
      const std::string& module_path, pybind11::module& m,
      const std::function<void(pybind11::module&)>& BuildModule);
};

}  // namespace nndeploy

#define NNDEPLOY_API_PYBIND11_MODULE(module_path, m)                \
  static void NndeployApiPythonModule##__LINE__(pybind11::module&); \
  namespace {                                                       \
  struct ApiRegistryInit {                                          \
    ApiRegistryInit() {                                             \
      ::nndeploy::NndeployModuleRegistry().Register(                \
          module_path, &NndeployApiPythonModule##__LINE__);         \
    }                                                               \
  };                                                                \
  ApiRegistryInit api_registry_init;                                \
  }                                                                 \
  static void NndeployApiPythonModule##__LINE__(pybind11::module& m)

#endif