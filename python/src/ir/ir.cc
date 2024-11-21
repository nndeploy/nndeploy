#include "nndeploy/ir/ir.h"

#include <pybind11/stl.h>

#include "nndeploy_api_registry.h"

namespace nndeploy {
namespace ir {
NNDEPLOY_API_PYBIND11_MODULE("ir", m) {
  py::class_<ir::ModelDesc>(m, "ModelDesc")
      .def(py::init<>())
      .def_readwrite("name_", &ir::ModelDesc::name_)
      .def(
          "weights",
          [](ModelDesc& self) {
            return py::cast(self.weights_, py::return_value_policy::reference);
          },
          py::return_value_policy::reference)
      // 定义如何设置weights_的值
      .def("setWeights", [](ModelDesc& self, const py::dict& weights) {
        for (const auto& kv : weights) {
          // 深拷贝
          const std::string& key = kv.first.cast<std::string>();
          device::Tensor* tensor = kv.second.cast<device::Tensor*>();

          self.weights_[key] = tensor->clone();
          self.weights_[key]->setName(key);
        }
      });
}
}  // namespace ir
}  // namespace nndeploy