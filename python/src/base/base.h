
#include "nndeploy/base/common.h"
#include "nndeploy/base/param.h"
#include "nndeploy/base/status.h"
#include "nndeploy/base/time_profiler.h"
#include "nndeploy/base/type.h"
#include "nndeploy_api_registry.h"

namespace nndeploy {
namespace base {

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
    PYBIND11_OVERRIDE_NAME(base::Status, BaseParam, "set", set, key, any);
  }

  base::Status get(const std::string &key, base::Any &any) override {
    PYBIND11_OVERRIDE_NAME(base::Status, BaseParam, "get", get, key, any);
  }

  base::Status serialize(rapidjson::Value &json,
                        rapidjson::Document::AllocatorType &allocator) override {
    PYBIND11_OVERRIDE_NAME(base::Status, BaseParam, "serialize", serialize, json, allocator);
  }

  base::Status serialize(std::ostream &stream) override {
    PYBIND11_OVERRIDE_NAME(base::Status, BaseParam, "serialize", serialize, stream);
  }

  base::Status serialize(std::string &content, bool is_file) override {
    PYBIND11_OVERRIDE_NAME(base::Status, BaseParam, "serialize", serialize, content, is_file);
  }

  base::Status deserialize(rapidjson::Value &json) override {
    PYBIND11_OVERRIDE_NAME(base::Status, BaseParam, "deserialize", deserialize, json);
  }

  base::Status deserialize(std::istream &stream) override {
    PYBIND11_OVERRIDE_NAME(base::Status, BaseParam, "deserialize", deserialize, stream);
  }

  base::Status deserialize(const std::string &content, bool is_file) override {
    PYBIND11_OVERRIDE_NAME(base::Status, BaseParam, "deserialize", deserialize, content, is_file);
  }
};

}  // namespace base
}  // namespace nndeploy
