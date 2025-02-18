#include "nndeploy/ir/ir.h"

#include <pybind11/stl.h>

#include "nndeploy_api_registry.h"

namespace nndeploy {
namespace ir {

NNDEPLOY_API_PYBIND11_MODULE("ir", m) {
  py::class_<OpDesc, std::shared_ptr<ir::OpDesc>>(m, "OpDesc")
      .def(py::init<>())
      .def(py::init<OpType>())
      .def(py::init<const std::string &, OpType>())
      .def(
          py::init<const std::string &, OpType, std::shared_ptr<base::Param>>())
      .def(py::init<const std::string &, OpType,
                    std::initializer_list<std::string>,
                    std::initializer_list<std::string>>())
      .def(py::init<
           const std::string &, OpType, std::initializer_list<std::string>,
           std::initializer_list<std::string>, std::shared_ptr<base::Param>>())
      .def(py::init<const std::string &, OpType, std::vector<std::string> &,
                    std::vector<std::string> &>())
      .def(py::init<const std::string &, OpType, std::vector<std::string> &,
                    std::vector<std::string> &, std::shared_ptr<base::Param>>())
      .def_readwrite("name_", &OpDesc::name_)
      .def_readwrite("op_type_", &OpDesc::op_type_)
      .def_readwrite("inputs_", &OpDesc::inputs_)
      .def_readwrite("outputs_", &OpDesc::outputs_)
      .def_readwrite("op_param_", &OpDesc::op_param_)
      .def("serialize", &OpDesc::serialize)
      .def("deserialize", &OpDesc::deserialize);

  py::class_<ir::ValueDesc, std::shared_ptr<ir::ValueDesc>>(m, "ValueDesc")
      .def(py::init<>())
      .def(py::init<const std::string &>())
      .def(py::init<const std::string &, base::DataType>())
      .def(py::init<const std::string &, base::DataType, base::IntVector>())
      .def_readwrite("name_", &ir::ValueDesc::name_)
      .def_readwrite("data_type_", &ir::ValueDesc::data_type_)
      .def_readwrite("shape_", &ir::ValueDesc::shape_)
      .def("serialize", &ir::ValueDesc::serialize)
      .def("deserialize", &ir::ValueDesc::deserialize);

  py::class_<ir::ModelDesc, std::shared_ptr<ir::ModelDesc>>(m, "ModelDesc")
      .def(py::init<>())
      .def_readwrite("name_", &ir::ModelDesc::name_)
      .def_readwrite("metadata_", &ir::ModelDesc::metadata_)
      .def_readwrite("inputs_", &ir::ModelDesc::inputs_)
      .def_readwrite("outputs_", &ir::ModelDesc::outputs_)
      .def_readwrite("op_descs_", &ir::ModelDesc::op_descs_)
      .def_readwrite("values_", &ir::ModelDesc::values_)
      .def(
          "weights",
          [](ModelDesc &self) {
            return py::cast(self.weights_, py::return_value_policy::reference);
          },
          py::return_value_policy::reference)
      // 定义如何设置weights_的值
      .def("set_weights",
           [](ModelDesc &self, const py::dict &weights) {
             for (const auto &kv : weights) {
               // 深拷贝
               const std::string &key = kv.first.cast<std::string>();
               device::Tensor *tensor = kv.second.cast<device::Tensor *>();

               self.weights_[key] = tensor->clone();
               self.weights_[key]->setName(key);
             }
           })
      .def("dump", &ir::ModelDesc::dump)
      .def("serialize_structure_to_json",
           py::overload_cast<rapidjson::Value &,
                             rapidjson::Document::AllocatorType &>(
               &ir::ModelDesc::serializeStructureToJson, py::const_))
      .def("serialize_structure_to_json",
           py::overload_cast<std::ostream &>(
               &ir::ModelDesc::serializeStructureToJson, py::const_))
      .def("serialize_structure_to_json",
           py::overload_cast<const std::string &>(
               &ir::ModelDesc::serializeStructureToJson, py::const_))
      .def(
          "deserialize_structure_from_json",
          py::overload_cast<rapidjson::Value &, const std::vector<ValueDesc> &>(
              &ir::ModelDesc::deserializeStructureFromJson))
      .def("deserialize_structure_from_json",
           py::overload_cast<std::istream &, const std::vector<ValueDesc> &>(
               &ir::ModelDesc::deserializeStructureFromJson))
      .def("deserialize_structure_from_json",
           py::overload_cast<const std::string &,
                             const std::vector<ValueDesc> &>(
               &ir::ModelDesc::deserializeStructureFromJson))
      .def("serialize_weights_to_safetensors",
           py::overload_cast<std::shared_ptr<safetensors::safetensors_t> &>(
               &ir::ModelDesc::serializeWeightsToSafetensors, py::const_))
      .def("serialize_weights_to_safetensors",
           py::overload_cast<const std::string &>(
               &ir::ModelDesc::serializeWeightsToSafetensors, py::const_))
      .def("deserialize_weights_from_safetensors",
           &ir::ModelDesc::deserializeWeightsFromSafetensors);
}

}  // namespace ir
}  // namespace nndeploy