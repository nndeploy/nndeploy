#include "nndeploy/dag/node.h"

#include "nndeploy/base/param.h"
#include "nndeploy/dag/edge.h"
#include "nndeploy/dag/graph.h"
#include "nndeploy_api_registry.h"

namespace py = pybind11;
namespace nndeploy {
namespace dag {

template<typename Base = Node>
class PyNode : public Base {
 public:
  using Base::Base;  // 继承构造函数

  base::Status setDeviceType(base::DeviceType device_type) override {
    PYBIND11_OVERRIDE_NAME(base::Status, Base, "set_device_type", setDeviceType,
                           device_type);
  }

  base::DeviceType getDeviceType() override {
    PYBIND11_OVERRIDE_NAME(base::DeviceType, Base, "get_device_type",
                           getDeviceType);
  }

  //   base::Status setParam(base::Param *param) override {
  //     PYBIND11_OVERRIDE_NAME(base::Status, Base, "set_param", setParam,
  //     param);
  //   }

  base::Status setParamSharedPtr(std::shared_ptr<base::Param> param) override {
    PYBIND11_OVERRIDE_NAME(base::Status, Base, "set_param", setParamSharedPtr,
                           param);
  }

  //   base::Param *getParam() override {
  //     PYBIND11_OVERRIDE_NAME(base::Param *, Base, "get_param", getParam);
  //   }

  std::shared_ptr<base::Param> getParamSharedPtr() override {
    PYBIND11_OVERRIDE_NAME(std::shared_ptr<base::Param>, Base, "get_param",
                           getParamSharedPtr);
  }

  base::Status setExternalParam(
      const std::string &key, std::shared_ptr<base::Param> external_param) override {
    PYBIND11_OVERRIDE_NAME(base::Status, Base, "set_external_param",
                           setExternalParam, key, external_param);
  }

  std::shared_ptr<base::Param> getExternalParam(
      const std::string &key) override {
    PYBIND11_OVERRIDE_NAME(std::shared_ptr<base::Param>, Base, "get_external_param",
                           getExternalParam, key);
  }

  base::Status init() override {
    PYBIND11_OVERRIDE_NAME(base::Status, Base, "init", init);
  }

  base::Status deinit() override {
    PYBIND11_OVERRIDE_NAME(base::Status, Base, "deinit", deinit);
  }

  int64_t getMemorySize() override {
    PYBIND11_OVERRIDE_NAME(int64_t, Base, "get_memory_size", getMemorySize);
  }

  base::Status setMemory(device::Buffer *buffer) override {
    PYBIND11_OVERRIDE_NAME(base::Status, Base, "set_memory", setMemory, buffer);
  }

  base::EdgeUpdateFlag updateInput() override {
    PYBIND11_OVERRIDE_NAME(base::EdgeUpdateFlag, Base, "update_input",
                           updateInput);
  }

  base::Status run() override {
    PYBIND11_OVERRIDE_PURE_NAME(base::Status, Base, "run", run);
  }
  
  std::vector<Edge *> operator()(
      std::vector<Edge *> inputs,
      std::vector<std::string> outputs_name = std::vector<std::string>(),
      std::shared_ptr<base::Param> param = nullptr) override {
    PYBIND11_OVERRIDE_NAME(std::vector<Edge *>, Base, "operator()", operator(),
                          inputs, outputs_name, param);
  }
   
  std::vector<std::string> getRealOutputsName(
      std::vector<std::string> outputs_name) override {
    PYBIND11_OVERRIDE_NAME(std::vector<std::string>, Base, "get_real_outputs_name",
                          getRealOutputsName, outputs_name);
  }
};

template<typename Base = Graph>
class PyGraph : public Base {
 public:
  using Base::Base;  // 继承构造函数

  virtual base::Status init() override {
    PYBIND11_OVERRIDE_NAME(base::Status, Base, "init", init);
  }

  virtual base::Status deinit() override {
    PYBIND11_OVERRIDE_NAME(base::Status, Base, "deinit", deinit);
  }

  virtual base::Status run() override {
    PYBIND11_OVERRIDE_NAME(base::Status, Base, "run", run);
  }

  std::vector<Edge *> operator()(
      std::vector<Edge *> inputs,
      std::vector<std::string> outputs_name = std::vector<std::string>(),
      std::shared_ptr<base::Param> param = nullptr) override {
    PYBIND11_OVERRIDE_NAME(std::vector<Edge *>, Base, "operator()", operator(),
                          inputs, outputs_name, param);
  }
};

}  // namespace dag
}  // namespace nndeploy