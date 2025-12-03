#include "nndeploy/base/param.h"
#include "nndeploy/dag/base.h"
#include "nndeploy/dag/composite_node.h"
#include "nndeploy/dag/condition.h"
#include "nndeploy/dag/const_node.h"
#include "nndeploy/dag/edge.h"
#include "nndeploy/dag/graph.h"
#include "nndeploy/dag/loop.h"
#include "nndeploy/dag/node.h"
#include "nndeploy/dag/running_condition.h"
#include "nndeploy/dag/util.h"
#include "nndeploy_api_registry.h"

namespace py = pybind11;
namespace nndeploy {
namespace dag {

struct PyObjectWrapper {
  PyObject *obj;  // 指向Python对象的指针

  // 构造函数:接收一个Python对象指针,增加引用计数防止对象被销毁
  PyObjectWrapper(PyObject *o) : obj(o) {
    // py::gil_scoped_acquire acquire;
    Py_INCREF(obj);
    // py::gil_scoped_release release;
  }

  // 析构函数:减少引用计数,允许Python回收对象
  ~PyObjectWrapper() {
    py::gil_scoped_acquire acquire;
    Py_DECREF(obj);
    // py::gil_scoped_release release;
  }
};

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

  base::Status setParamSharedPtr(std::shared_ptr<base::Param> param) override {
    PYBIND11_OVERRIDE_NAME(base::Status, Base, "set_param", setParamSharedPtr,
                           param);
  }

  base::Status setParam(const std::string &key,
                        const std::string &value) override {
    PYBIND11_OVERRIDE_NAME(base::Status, Base, "set_param", setParam, key,
                           value);
  }

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

  base::Status addResourceWithoutState(const std::string &key,
                                       const base::Any &value) override {
    PYBIND11_OVERRIDE_NAME(base::Status, Base, "add_resource_without_state",
                           addResourceWithoutState, key, value);
  }

  base::Any &getResourceWithoutState(const std::string &key) override {
    PYBIND11_OVERRIDE_NAME(base::Any &, Base, "get_resource_without_state",
                           getResourceWithoutState, key);
  }

  Edge *createResourceWithState(const std::string &key) override {
    PYBIND11_OVERRIDE_NAME(Edge *, Base, "create_resource_with_state",
                           createResourceWithState, key);
  }

  base::Status addResourceWithState(const std::string &key,
                                    Edge *edge) override {
    PYBIND11_OVERRIDE_NAME(base::Status, Base, "add_resource_with_state",
                           addResourceWithState, key, edge);
  }

  Edge *getResourceWithState(const std::string &key) override {
    PYBIND11_OVERRIDE_NAME(Edge *, Base, "get_resource_with_state",
                           getResourceWithState, key);
  }

  base::Status setInput(Edge *input, int index = -1) override {
    PYBIND11_OVERRIDE_NAME(base::Status, Base, "set_input", setInput, input,
                           index);
  }

  base::Status setOutput(Edge *output, int index = -1) override {
    PYBIND11_OVERRIDE_NAME(base::Status, Base, "set_output", setOutput, output,
                           index);
  }

  base::Status setIterInput(Edge *input, int index = -1) override {
    PYBIND11_OVERRIDE_NAME(base::Status, Base, "set_iter_input", setIterInput,
                           input, index);
  }

  base::Status setInputs(std::vector<Edge *> inputs) override {
    PYBIND11_OVERRIDE_NAME(base::Status, Base, "set_inputs", setInputs, inputs);
  }

  base::Status setOutputs(std::vector<Edge *> outputs) override {
    PYBIND11_OVERRIDE_NAME(base::Status, Base, "set_outputs", setOutputs,
                           outputs);
  }

  Edge *createInternalOutputEdge(const std::string &name) override {
    PYBIND11_OVERRIDE_NAME(Edge *, Base, "create_internal_output_edge",
                           createInternalOutputEdge, name);
  }

  base::Status setParallelType(
      const base::ParallelType &parallel_type) override {
    PYBIND11_OVERRIDE_NAME(base::Status, Base, "set_parallel_type",
                           setParallelType, parallel_type);
  }

  base::ParallelType getParallelType() override {
    PYBIND11_OVERRIDE_NAME(base::ParallelType, Base, "get_parallel_type",
                           getParallelType);
  }

  std::shared_ptr<RunStatus> getRunStatus() override {
    PYBIND11_OVERRIDE_NAME(std::shared_ptr<RunStatus>, Base, "get_run_status",
                           getRunStatus);
  }

  virtual void setTraceFlag(bool flag) override {
    PYBIND11_OVERRIDE_NAME(void, Base, "set_trace_flag", setTraceFlag, flag);
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

  bool synchronize() override {
    PYBIND11_OVERRIDE_NAME(bool, Base, "synchronize", synchronize);
  }

  bool interrupt() override {
    PYBIND11_OVERRIDE_NAME(bool, Base, "interrupt", interrupt);
  }

  bool checkInterruptStatus() override {
    PYBIND11_OVERRIDE_NAME(bool, Base, "check_interrupt_status",
                           checkInterruptStatus);
  }

  void clearInterrupt() override {
    PYBIND11_OVERRIDE_NAME(void, Base, "clear_interrupt", clearInterrupt);
  }

  std::vector<Edge *> forward(std::vector<Edge *> inputs) override {
    PYBIND11_OVERRIDE_NAME(std::vector<Edge *>, Base, "forward", forward,
                           inputs);
  }

  std::vector<Edge *> operator()(std::vector<Edge *> inputs) override {
    PYBIND11_OVERRIDE_NAME(std::vector<Edge *>, Base, "operator()", operator(),
                           inputs);
  }

  std::vector<Edge *> forward() override {
    PYBIND11_OVERRIDE_NAME(std::vector<Edge *>, Base, "forward", forward);
  }

  std::vector<Edge *> operator()() override {
    PYBIND11_OVERRIDE_NAME(std::vector<Edge *>, Base, "operator()", operator());
  }

  std::vector<Edge *> forward(Edge *input) override {
    PYBIND11_OVERRIDE_NAME(std::vector<Edge *>, Base, "forward", forward,
                           input);
  }

  std::vector<Edge *> operator()(Edge *input) override {
    PYBIND11_OVERRIDE_NAME(std::vector<Edge *>, Base, "operator()", operator(),
                           input);
  }

  base::Status toStaticGraph() override {
    PYBIND11_OVERRIDE_NAME(base::Status, Base, "to_static_graph",
                           toStaticGraph);
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

  virtual void setLoopCount(int loop_count) override {
    PYBIND11_OVERRIDE_NAME(void, Base, "set_loop_count", setLoopCount,
                           loop_count);
  }

  virtual int getLoopCount() override {
    PYBIND11_OVERRIDE_NAME(int, Base, "get_loop_count", getLoopCount);
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

  virtual base::Status setInput(Edge *input, int index = -1) override {
    PYBIND11_OVERRIDE_NAME(base::Status, Base, "set_input", setInput, input,
                           index);
  }

  virtual base::Status setOutput(Edge *output, int index = -1) override {
    PYBIND11_OVERRIDE_NAME(base::Status, Base, "set_output", setOutput, output,
                           index);
  }

  virtual base::Status setInputs(std::vector<Edge *> inputs) override {
    PYBIND11_OVERRIDE_NAME(base::Status, Base, "set_inputs", setInputs, inputs);
  }

  virtual base::Status setOutputs(std::vector<Edge *> outputs) override {
    PYBIND11_OVERRIDE_NAME(base::Status, Base, "set_outputs", setOutputs,
                           outputs);
  }

  virtual base::Status setInputSharedPtr(std::shared_ptr<Edge> input,
                                         int index = -1) override {
    PYBIND11_OVERRIDE_NAME(base::Status, Base, "set_input_shared_ptr",
                           setInputSharedPtr, input, index);
  }

  virtual base::Status setOutputSharedPtr(std::shared_ptr<Edge> output,
                                          int index = -1) override {
    PYBIND11_OVERRIDE_NAME(base::Status, Base, "set_output_shared_ptr",
                           setOutputSharedPtr, output, index);
  }

  virtual base::Status setInputsSharedPtr(
      std::vector<std::shared_ptr<Edge>> inputs) override {
    PYBIND11_OVERRIDE_NAME(base::Status, Base, "set_inputs_shared_ptr",
                           setInputsSharedPtr, inputs);
  }

  virtual base::Status setOutputsSharedPtr(
      std::vector<std::shared_ptr<Edge>> outputs) override {
    PYBIND11_OVERRIDE_NAME(base::Status, Base, "set_outputs_shared_ptr",
                           setOutputsSharedPtr, outputs);
  }

  virtual void setLoopMaxFlag(bool is_loop_max_flag) override {
    PYBIND11_OVERRIDE_NAME(void, Base, "set_loop_max_flag", setLoopMaxFlag,
                           is_loop_max_flag);
  }

  virtual bool getLoopMaxFlag() override {
    PYBIND11_OVERRIDE_NAME(bool, Base, "get_loop_max_flag", getLoopMaxFlag);
  }

  virtual void setLoopCount(int loop_count) override {
    PYBIND11_OVERRIDE_NAME(void, Base, "set_loop_count", setLoopCount,
                           loop_count);
  }

  virtual int getLoopCount() override {
    PYBIND11_OVERRIDE_NAME(int, Base, "get_loop_count", getLoopCount);
  }

  // 在类定义前添加类型别名
  using LoopCountMap = std::map<std::string, int>;

  // 然后在方法中使用
  virtual LoopCountMap getLoopCountMap() override {
    PYBIND11_OVERRIDE_NAME(LoopCountMap, Base, "get_loop_count_map",
                           getLoopCountMap);
  }

  virtual base::Status defaultParam() override {
    PYBIND11_OVERRIDE_NAME(base::Status, Base, "default_param", defaultParam);
  }

  virtual base::Status init() override {
    PYBIND11_OVERRIDE_NAME(base::Status, Base, "init", init);
  }

  virtual base::Status deinit() override {
    PYBIND11_OVERRIDE_NAME(base::Status, Base, "deinit", deinit);
  }

  virtual base::Status run() override {
    PYBIND11_OVERRIDE_NAME(base::Status, Base, "run", run);
  }

  virtual bool synchronize() override {
    PYBIND11_OVERRIDE_NAME(bool, Base, "synchronize", synchronize);
  }

  virtual bool interrupt() override {
    PYBIND11_OVERRIDE_NAME(bool, Base, "interrupt", interrupt);
  }

  std::vector<Edge *> forward(std::vector<Edge *> inputs) override {
    PYBIND11_OVERRIDE_NAME(std::vector<Edge *>, Base, "forward", forward,
                           inputs);
  }

  std::vector<Edge *> operator()(std::vector<Edge *> inputs) override {
    PYBIND11_OVERRIDE_NAME(std::vector<Edge *>, Base, "operator()", operator(),
                           inputs);
  }

  std::vector<Edge *> forward() override {
    PYBIND11_OVERRIDE_NAME(std::vector<Edge *>, Base, "forward", forward);
  }

  std::vector<Edge *> operator()() override {
    PYBIND11_OVERRIDE_NAME(std::vector<Edge *>, Base, "operator()", operator());
  }

  std::vector<Edge *> forward(Edge *input) override {
    PYBIND11_OVERRIDE_NAME(std::vector<Edge *>, Base, "forward", forward,
                           input);
  }

  std::vector<Edge *> operator()(Edge *input) override {
    PYBIND11_OVERRIDE_NAME(std::vector<Edge *>, Base, "operator()", operator(),
                           input);
  }

  virtual void setTraceFlag(bool flag) override {
    PYBIND11_OVERRIDE_NAME(void, Base, "set_trace_flag", setTraceFlag, flag);
  }

  base::Status toStaticGraph() override {
    PYBIND11_OVERRIDE_NAME(base::Status, Base, "to_static_graph",
                           toStaticGraph);
  }

  virtual base::Status addResourceWithoutState(
      const std::string &key, const base::Any &value) override {
    PYBIND11_OVERRIDE_NAME(base::Status, Base, "add_resource_without_state",
                           addResourceWithoutState, key, value);
  }

  virtual base::Any &getResourceWithoutState(const std::string &key) override {
    PYBIND11_OVERRIDE_NAME(base::Any &, Base, "get_resource_without_state",
                           getResourceWithoutState, key);
  }

  virtual base::Status addResourceWithState(const std::string &key,
                                            Edge *edge) override {
    PYBIND11_OVERRIDE_NAME(base::Status, Base, "add_resource_with_state",
                           addResourceWithState, key, edge);
  }

  virtual Edge *getResourceWithState(const std::string &key) override {
    PYBIND11_OVERRIDE_NAME(Edge *, Base, "get_resource_with_state",
                           getResourceWithState, key);
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

  virtual void setUnusedNodeNames(const std::string &node_name) override {
    PYBIND11_OVERRIDE_NAME(void, Base, "set_unused_node_names",
                           setUnusedNodeNames, node_name);
  }

  virtual void setUnusedNodeNames(
      const std::set<std::string> &node_names) override {
    PYBIND11_OVERRIDE_NAME(void, Base, "set_unused_node_names",
                           setUnusedNodeNames, node_names);
  }

  virtual void removeUnusedNodeNames(const std::string &node_name) override {
    PYBIND11_OVERRIDE_NAME(void, Base, "remove_unused_node_names",
                           removeUnusedNodeNames, node_name);
  }

  virtual void removeUnusedNodeNames(
      const std::set<std::string> &node_names) override {
    PYBIND11_OVERRIDE_NAME(void, Base, "remove_unused_node_names",
                           removeUnusedNodeNames, node_names);
  }

  virtual std::set<std::string> getUnusedNodeNames() override {
    PYBIND11_OVERRIDE_NAME(std::set<std::string>, Base, "get_unused_node_names",
                           getUnusedNodeNames);
  }

  virtual void removeInOutNode() override {
    PYBIND11_OVERRIDE_NAME(void, Base, "remove_in_out_node", removeInOutNode);
  }

  virtual void setNodeValue(const std::string &node_value_str) override {
    PYBIND11_OVERRIDE_NAME(void, Base, "set_node_value", setNodeValue,
                           node_value_str);
  }

  virtual void setNodeValue(const std::string &node_name,
                            const std::string &key,
                            const std::string &value) override {
    PYBIND11_OVERRIDE_NAME(void, Base, "set_node_value", setNodeValue,
                           node_name, key, value);
  }

  virtual void setNodeValue(
      std::map<std::string, std::map<std::string, std::string>> node_value_map)
      override {
    PYBIND11_OVERRIDE_NAME(void, Base, "set_node_value", setNodeValue,
                           node_value_map);
  }

  virtual std::map<std::string, std::map<std::string, std::string>>
  getNodeValue() override {
    using ReturnType =
        std::map<std::string, std::map<std::string, std::string>>;
    PYBIND11_OVERRIDE_NAME(ReturnType, Base, "get_node_value", getNodeValue);
  }
};

template <typename Base = ConstNode>
class PyConstNode : public Base {
 public:
  using Base::Base;

  base::EdgeUpdateFlag updateInput() override {
    PYBIND11_OVERRIDE_PURE_NAME(base::EdgeUpdateFlag, ConstNode, "update_input",
                                updateInput);
  }

  base::Status init() override {
    PYBIND11_OVERRIDE_NAME(base::Status, ConstNode, "init", init);
  }

  base::Status deinit() override {
    PYBIND11_OVERRIDE_NAME(base::Status, ConstNode, "deinit", deinit);
  }

  base::Status run() override {
    PYBIND11_OVERRIDE_PURE_NAME(base::Status, ConstNode, "run", run);
  }
};

template <typename Base = CompositeNode>
class PyCompositeNode : public Base {
 public:
  using Base::Base;

  base::Status setInput(Edge *input, int index = -1) override {
    PYBIND11_OVERRIDE_NAME(base::Status, CompositeNode, "set_input", setInput,
                           input, index);
  }

  base::Status setOutput(Edge *output, int index = -1) override {
    PYBIND11_OVERRIDE_NAME(base::Status, CompositeNode, "set_output", setOutput,
                           output, index);
  }

  base::Status setInputs(std::vector<Edge *> inputs) override {
    PYBIND11_OVERRIDE_NAME(base::Status, CompositeNode, "set_inputs", setInputs,
                           inputs);
  }

  base::Status setOutputs(std::vector<Edge *> outputs) override {
    PYBIND11_OVERRIDE_NAME(base::Status, CompositeNode, "set_outputs",
                           setOutputs, outputs);
  }

  // base::Status setInputSharedPtr(std::shared_ptr<Edge> input, int index = -1)
  // override {
  //   PYBIND11_OVERRIDE_NAME(base::Status, CompositeNode,
  //   "set_input_shared_ptr", setInputSharedPtr, input, index);
  // }

  // base::Status setOutputSharedPtr(std::shared_ptr<Edge> output, int index =
  // -1) override {
  //   PYBIND11_OVERRIDE_NAME(base::Status, CompositeNode,
  //   "set_output_shared_ptr", setOutputSharedPtr, output, index);
  // }

  // base::Status setInputsSharedPtr(std::vector<std::shared_ptr<Edge>> inputs)
  // override {
  //   PYBIND11_OVERRIDE_NAME(base::Status, CompositeNode,
  //   "set_inputs_shared_ptr", setInputsSharedPtr, inputs);
  // }

  // base::Status setOutputsSharedPtr(std::vector<std::shared_ptr<Edge>>
  // outputs) override {
  //   PYBIND11_OVERRIDE_NAME(base::Status, CompositeNode,
  //   "set_outputs_shared_ptr", setOutputsSharedPtr, outputs);
  // }

  base::Status defaultParam() override {
    PYBIND11_OVERRIDE_NAME(base::Status, CompositeNode, "default_param",
                           defaultParam);
  }

  base::Status init() override {
    PYBIND11_OVERRIDE_NAME(base::Status, CompositeNode, "init", init);
  }

  base::Status deinit() override {
    PYBIND11_OVERRIDE_NAME(base::Status, CompositeNode, "deinit", deinit);
  }

  base::Status run() override {
    PYBIND11_OVERRIDE_PURE_NAME(base::Status, CompositeNode, "run", run);
  }

  base::Status serialize(
      rapidjson::Value &json,
      rapidjson::Document::AllocatorType &allocator) override {
    PYBIND11_OVERRIDE_NAME(base::Status, CompositeNode, "serialize", serialize,
                           json, allocator);
  }

  std::string serialize() override {
    PYBIND11_OVERRIDE_NAME(std::string, CompositeNode, "serialize", serialize);
  }

  base::Status deserialize(rapidjson::Value &json) override {
    PYBIND11_OVERRIDE_NAME(base::Status, CompositeNode, "deserialize",
                           deserialize, json);
  }

  base::Status deserialize(const std::string &json_str) override {
    PYBIND11_OVERRIDE_NAME(base::Status, CompositeNode, "deserialize",
                           deserialize, json_str);
  }
};

template <typename Base = Condition>
class PyCondition : public Base {
 public:
  using Base::Base;

  base::Status init() override {
    PYBIND11_OVERRIDE_NAME(base::Status, Condition, "init", init);
  }

  base::Status deinit() override {
    PYBIND11_OVERRIDE_NAME(base::Status, Condition, "deinit", deinit);
  }

  int choose() override {
    PYBIND11_OVERRIDE_PURE_NAME(int, Condition, "choose", choose);
  }

  base::Status run() override {
    PYBIND11_OVERRIDE_NAME(base::Status, Condition, "run", run);
  }
};

template <typename Base = Loop>
class PyLoop : public Base {
 public:
  using Base::Base;

  base::Status init() override {
    PYBIND11_OVERRIDE_NAME(base::Status, Loop, "init", init);
  }

  base::Status deinit() override {
    PYBIND11_OVERRIDE_NAME(base::Status, Loop, "deinit", deinit);
  }

  int loops() override {
    PYBIND11_OVERRIDE_PURE_NAME(int, Loop, "loops", loops);
  }

  base::Status run() override {
    PYBIND11_OVERRIDE_NAME(base::Status, Loop, "run", run);
  }

  // base::Status serialize(rapidjson::Value &json,
  // rapidjson::Document::AllocatorType &allocator) override {
  //   PYBIND11_OVERRIDE_NAME(base::Status, Loop, "serialize", serialize, json,
  //   allocator);
  // }

  // std::string serialize() override {
  //   PYBIND11_OVERRIDE_NAME(std::string, Loop, "serialize", serialize);
  // }
};

template <typename Base = RunningCondition>
class PyRunningCondition : public Base {
 public:
  using Base::Base;

  int choose() override {
    PYBIND11_OVERRIDE_PURE_NAME(int, RunningCondition, "choose", choose);
  }
};

}  // namespace dag
}  // namespace nndeploy