
#include "nndeploy/base/common.h"
#include "nndeploy/base/param.h"
#include "nndeploy/base/status.h"
#include "nndeploy/base/time_profiler.h"
#include "nndeploy/base/type.h"
#include "nndeploy_api_registry.h"

namespace nndeploy {
namespace base {

inline Any pythonToAny(py::handle value) {
  Any any;
  if (py::isinstance<py::bool_>(value)) any = value.cast<bool>();
  else if (py::isinstance<py::int_>(value)) any = value.cast<int>();
  else if (py::isinstance<py::float_>(value)) any = value.cast<double>();
  else if (py::isinstance<py::str>(value)) any = value.cast<std::string>();
  else if (py::isinstance<py::sequence>(value)) any = value.cast<std::vector<std::string>>();
  else throw py::type_error("Unsupported parameter type");
  return any;
}

inline py::object anyToPython(const Any &any) {
  if (any.type() == typeid(bool)) return py::cast(get<bool>(any));
  if (any.type() == typeid(int)) return py::cast(get<int>(any));
  if (any.type() == typeid(double)) return py::cast(get<double>(any));
  if (any.type() == typeid(std::string)) return py::cast(get<std::string>(any));
  if (any.type() == typeid(std::vector<std::string>))
    return py::cast(get<std::vector<std::string>>(any));
  throw py::type_error("Unsupported parameter type");
}

template <typename BaseParam = Param>
class PyParam : public BaseParam {
 public:
  using BaseParam::BaseParam;

  std::shared_ptr<nndeploy::base::Param> copy() override {
    PYBIND11_OVERRIDE_NAME(std::shared_ptr<nndeploy::base::Param>, BaseParam, "copy", copy);
  }

  base::Status copyTo(nndeploy::base::Param *param) override {
    PYBIND11_OVERRIDE_NAME(base::Status, BaseParam, "copy_to", copyTo, param);
  }

  base::Status set(const std::string &key, base::Any &any) override {
    py::gil_scoped_acquire gil;
    auto override = py::get_override(static_cast<const BaseParam *>(this), "set");
    if (override) return override(key, anyToPython(any)).template cast<base::Status>();
    return BaseParam::set(key, any);
  }

  base::Status get(const std::string &key, base::Any &any) override {
    py::gil_scoped_acquire gil;
    auto override = py::get_override(static_cast<const BaseParam *>(this), "get");
    if (!override) return BaseParam::get(key, any);
    any = pythonToAny(override(key));
    return kStatusCodeOk;
  }

  base::Status serialize(rapidjson::Value &json,
                        rapidjson::Document::AllocatorType &allocator) override {
    PYBIND11_OVERRIDE_NAME(base::Status, BaseParam, "serialize", serialize, json, allocator);
  }

  std::string serialize() override {
    PYBIND11_OVERRIDE_NAME(std::string, BaseParam, "serialize", serialize);
  }

  base::Status saveFile(const std::string &path) override {
    PYBIND11_OVERRIDE_NAME(base::Status, BaseParam, "save_file", saveFile, path);
  }

  base::Status deserialize(rapidjson::Value &json) override {
    PYBIND11_OVERRIDE_NAME(base::Status, BaseParam, "deserialize", deserialize, json);
  }

  base::Status deserialize(const std::string &json_str) override {
    PYBIND11_OVERRIDE_NAME(base::Status, BaseParam, "deserialize", deserialize, json_str);
  }

  base::Status loadFile(const std::string &path) override {
    PYBIND11_OVERRIDE_NAME(base::Status, BaseParam, "load_file", loadFile, path);
  }
};

}  // namespace base
}  // namespace nndeploy
