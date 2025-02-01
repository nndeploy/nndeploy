#include "nndeploy/ir/ir.h"

#include <pybind11/stl.h>

#include "nndeploy_api_registry.h"

namespace nndeploy {
namespace ir {
NNDEPLOY_API_PYBIND11_MODULE("ir", m) {
  py::class_<ir::ValueDesc, std::shared_ptr<ir::ValueDesc>>(m, "ValueDesc")
      .def(py::init<>())
      .def(py::init<const std::string&>())
      .def(py::init<const std::string&, base::DataType>())
      .def(py::init<const std::string&, base::DataType, base::IntVector>())
      .def_readwrite("name_", &ir::ValueDesc::name_)
      .def_readwrite("type_", &ir::ValueDesc::data_type_)
      .def_readwrite("shape_", &ir::ValueDesc::shape_);

  py::class_<ir::ModelDesc, std::shared_ptr<ir::ModelDesc>>(m, "ModelDesc")
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

  py::class_<OpDesc, std::shared_ptr<ir::OpDesc>>(m, "OpDesc");
}
}  // namespace ir
}  // namespace nndeploy