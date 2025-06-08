#include "nndeploy/base/param.h"
#include "nndeploy/dag/edge.h"
#include "nndeploy/dag/graph.h"
#include "nndeploy/dag/node.h"
#include "nndeploy_api_registry.h"

namespace py = pybind11;
namespace nndeploy {
namespace dag {

template <typename Base = Node>
class PyNode : public Base {
 public:
  using Base::Base;  // 继承构造函数

  base::Status setInputName(const std::string &name, int index) override {
    PYBIND11_OVERRIDE_NAME(base::Status, Base, "set_input_name", setInputName,
                           name, index);
  }

  base::Status setOutputName(const std::string &name, int index) override {
    PYBIND11_OVERRIDE_NAME(base::Status, Base, "set_output_name", setOutputName,
                           name, index);
  }

  base::Status setInputNames(const std::vector<std::string> &names) override {
    PYBIND11_OVERRIDE_NAME(base::Status, Base, "set_input_names", setInputNames,
                           names);
  }

  base::Status setOutputNames(const std::vector<std::string> &names) override {
    PYBIND11_OVERRIDE_NAME(base::Status, Base, "set_output_names",
                           setOutputNames, names);
  }

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
      const std::string &key,
      std::shared_ptr<base::Param> external_param) override {
    PYBIND11_OVERRIDE_NAME(base::Status, Base, "set_external_param",
                           setExternalParam, key, external_param);
  }

  std::shared_ptr<base::Param> getExternalParam(
      const std::string &key) override {
    PYBIND11_OVERRIDE_NAME(std::shared_ptr<base::Param>, Base,
                           "get_external_param", getExternalParam, key);
  }

  base::Status defaultParam() override {
    PYBIND11_OVERRIDE_NAME(base::Status, Base, "default_param", defaultParam);
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

  std::vector<Edge *> forward(std::vector<Edge *> inputs) override {
    PYBIND11_OVERRIDE_NAME(std::vector<Edge *>, Base, "forward", forward,
                           inputs);
  }

  std::vector<Edge *> operator()(std::vector<Edge *> inputs) override {
    PYBIND11_OVERRIDE_NAME(std::vector<Edge *>, Base, "operator()", operator(),
                           inputs);
  }

  std::vector<std::string> getRealOutputsName() override {
    PYBIND11_OVERRIDE_NAME(std::vector<std::string>, Base,
                           "get_real_outputs_name", getRealOutputsName);
  }

  base::Status serialize(
      rapidjson::Value &json,
      rapidjson::Document::AllocatorType &allocator) override {
    PYBIND11_OVERRIDE_NAME(base::Status, Base, "serialize", serialize, json,
                           allocator);
  }

  std::string serialize() override {
    PYBIND11_OVERRIDE_NAME(std::string, Base, "serialize", serialize);
  }

  base::Status saveFile(const std::string &path) override {
    PYBIND11_OVERRIDE_NAME(base::Status, Base, "save_file", saveFile, path);
  }

  base::Status deserialize(rapidjson::Value &json) override {
    PYBIND11_OVERRIDE_NAME(base::Status, Base, "deserialize", deserialize,
                           json);
  }

  base::Status deserialize(const std::string &json_str) override {
    PYBIND11_OVERRIDE_NAME(base::Status, Base, "deserialize", deserialize,
                           json_str);
  }

  base::Status loadFile(const std::string &path) override {
    PYBIND11_OVERRIDE_NAME(base::Status, Base, "load_file", loadFile, path);
  }
};

template <typename Base = NodeCreator>
class PyNodeCreator : public Base {
 public:
  using Base::Base;

  Node *createNode(const std::string &node_name, std::vector<Edge *> inputs,
                   std::vector<Edge *> outputs) override {
    PYBIND11_OVERRIDE_PURE_NAME(Node *, NodeCreator, "create_node", createNode,
                                node_name, inputs, outputs);
  }

  std::shared_ptr<Node> createNodeSharedPtr(
      const std::string &node_name, std::vector<Edge *> inputs,
      std::vector<Edge *> outputs) override {
    PYBIND11_OVERRIDE_PURE_NAME(std::shared_ptr<Node>, NodeCreator,
                                "create_node_shared_ptr", createNodeSharedPtr,
                                node_name, inputs, outputs);
  }
};

template <typename Base = Graph>
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

  std::vector<Edge *> forward(std::vector<Edge *> inputs) override {
    PYBIND11_OVERRIDE_NAME(std::vector<Edge *>, Base, "forward", forward,
                           inputs);
  }

  std::vector<Edge *> operator()(std::vector<Edge *> inputs) override {
    PYBIND11_OVERRIDE_NAME(std::vector<Edge *>, Base, "operator()", operator(),
                           inputs);
  }

  virtual base::Status serialize(
      rapidjson::Value &json,
      rapidjson::Document::AllocatorType &allocator) override {
    PYBIND11_OVERRIDE_NAME(base::Status, Base, "serialize", serialize, json,
                           allocator);
  }

  virtual std::string serialize() override {
    PYBIND11_OVERRIDE_NAME(std::string, Base, "serialize", serialize);
  }

  virtual base::Status deserialize(rapidjson::Value &json) override {
    PYBIND11_OVERRIDE_NAME(base::Status, Base, "deserialize", deserialize,
                           json);
  }

  virtual base::Status deserialize(const std::string &json_str) override {
    PYBIND11_OVERRIDE_NAME(base::Status, Base, "deserialize", deserialize,
                           json_str);
  }
};

}  // namespace dag
}  // namespace nndeploy