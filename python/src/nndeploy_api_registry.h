#ifndef _NNDEPLOY_PYTHON_SRC_NNDEPLOY_API_REGISTRY_h_
#define _NNDEPLOY_PYTHON_SRC_NNDEPLOY_API_REGISTRY_h_

// pybind11核心功能头文件
#include <pybind11/pybind11.h>
// STL绑定支持
#include <pybind11/stl.h>
// std::function和lambda表达式绑定支持
#include <pybind11/functional.h>
// 时间相关类型绑定支持，如std::chrono
#include <pybind11/chrono.h>
// 复数类型绑定支持，如std::complex
#include <pybind11/complex.h>
// Eigen库矩阵类型绑定支持
// #include <pybind11/eigen.h>
// 嵌入Python解释器支持
#include <pybind11/embed.h>
// 在C++中执行Python代码支持
#include <pybind11/eval.h>
// 重定向Python输入输出流支持，如sys.stdout
#include <pybind11/iostream.h>
// Numpy多维数组绑定支持
#include <pybind11/numpy.h>
// 运算符重载绑定支持
#include <pybind11/operators.h>
// Python基本类型转换支持
#include <pybind11/pytypes.h>
// 字符串流支持
#include <pybind11/iostream.h>

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