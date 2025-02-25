#include "nndeploy/dag/edge.h"
#include "nndeploy/dag/graph.h"
#include "nndeploy/dag/node.h"

#include "nndeploy/dag/edge.h"
#include "nndeploy_api_registry.h"

namespace py = pybind11;
namespace nndeploy {
namespace dag {

class PyNode : public Node {
 public:
  using Node::Node;  // 继承构造函数

  base::Status setDeviceType(base::DeviceType device_type) override {
    PYBIND11_OVERRIDE(base::Status, Node, setDeviceType, device_type);
  }

  base::DeviceType getDeviceType() override {
    PYBIND11_OVERRIDE(base::DeviceType, Node, getDeviceType);
  }

  base::Status setParam(base::Param *param) override {
    PYBIND11_OVERRIDE(base::Status, Node, setParam, param);
  }

  base::Status setParamSharedPtr(std::shared_ptr<base::Param> param) override {
    PYBIND11_OVERRIDE(base::Status, Node, setParamSharedPtr, param);
  }

  base::Param *getParam() override {
    PYBIND11_OVERRIDE(base::Param*, Node, getParam);
  }

  std::shared_ptr<base::Param> getParamSharedPtr() override {
    PYBIND11_OVERRIDE(std::shared_ptr<base::Param>, Node, getParamSharedPtr);
  }

  base::Status setExternalParam(base::Param *external_param) override {
    PYBIND11_OVERRIDE(base::Status, Node, setExternalParam, external_param);
  }

  base::Status setExternalParamSharedPtr(std::shared_ptr<base::Param> external_param) override {
    PYBIND11_OVERRIDE(base::Status, Node, setExternalParamSharedPtr, external_param);
  }

  base::Status init() override {
    PYBIND11_OVERRIDE(base::Status, Node, init);
  }

  base::Status deinit() override {
    PYBIND11_OVERRIDE(base::Status, Node, deinit);
  }

  int64_t getMemorySize() override {
    PYBIND11_OVERRIDE(int64_t, Node, getMemorySize);
  }

  base::Status setMemory(device::Buffer *buffer) override {
    PYBIND11_OVERRIDE(base::Status, Node, setMemory, buffer);
  }

  base::EdgeUpdateFlag updataInput() override {
    PYBIND11_OVERRIDE(base::EdgeUpdateFlag, Node, updataInput);
  }

  base::Status run() override {
    PYBIND11_OVERRIDE_PURE(base::Status, Node, run);
  }

  std::vector<std::shared_ptr<Edge>> operator()(
      std::vector<std::shared_ptr<Edge>> inputs,
      std::vector<std::string> outputs_name,
      std::shared_ptr<base::Param> param) override {
    PYBIND11_OVERRIDE(std::vector<std::shared_ptr<Edge>>, Node, operator(), inputs, outputs_name, param);
  }

  std::vector<std::shared_ptr<Edge>> functorWithoutGraph(
      std::vector<std::shared_ptr<Edge>> inputs,
      std::vector<std::string> outputs_name,
      std::shared_ptr<base::Param> param) override {
    PYBIND11_OVERRIDE(std::vector<std::shared_ptr<Edge>>, Node, functorWithoutGraph, inputs, outputs_name, param);
  }

  std::vector<std::shared_ptr<Edge>> functorWithGraph(
      std::vector<std::shared_ptr<Edge>> inputs,
      std::vector<std::string> outputs_name,
      std::shared_ptr<base::Param> param) override {
    PYBIND11_OVERRIDE(std::vector<std::shared_ptr<Edge>>, Node, functorWithGraph, inputs, outputs_name, param);
  }

  std::vector<std::shared_ptr<Edge>> functorDynamic(
      std::vector<std::shared_ptr<Edge>> inputs,
      std::vector<std::string> outputs_name,
      std::shared_ptr<base::Param> param) override {
    PYBIND11_OVERRIDE(std::vector<std::shared_ptr<Edge>>, Node, functorDynamic, inputs, outputs_name, param);
  }
};


NNDEPLOY_API_PYBIND11_MODULE("dag", m) {
  py::class_<NodeDesc, std::shared_ptr<NodeDesc>>(m, "NodeDesc", py::dynamic_attr())
      .def(py::init<const std::string&, std::initializer_list<std::string>, std::initializer_list<std::string>>())
      .def(py::init<const std::string&, std::vector<std::string>, std::vector<std::string>>())
      .def(py::init<const std::string&, const std::string&, std::initializer_list<std::string>, std::initializer_list<std::string>>())
      .def(py::init<const std::string&, const std::string&, std::vector<std::string>, std::vector<std::string>>())
      .def("get_key", &NodeDesc::getKey)
      .def("get_name", &NodeDesc::getName)
      .def("get_inputs", &NodeDesc::getInputs)
      .def("get_outputs", &NodeDesc::getOutputs);

  // 定义Node类绑定
  py::class_<Node, PyNode, std::shared_ptr<Node>>(m, "Node", py::dynamic_attr())
      // 构造函数
      .def(py::init<const std::string&>())
      .def(py::init<const std::string&, Edge*, Edge*>())
      .def(py::init<const std::string&, std::initializer_list<Edge*>,
                    std::initializer_list<Edge*>>())
      .def(py::init<const std::string&, std::vector<Edge*>,
                    std::vector<Edge*>>())

      // 基本属性访问
      .def("get_name", &Node::getName)
      .def("set_graph", &Node::setGraph)
      .def("get_graph", &Node::getGraph)
      .def("set_device_type", &Node::setDeviceType)
      .def("get_device_type", &Node::getDeviceType)

      // 参数相关
      .def("set_param", &Node::setParam)
      .def("set_param_shared_ptr", &Node::setParamSharedPtr)
      .def("get_param", &Node::getParam)
      .def("get_param_shared_ptr", &Node::getParamSharedPtr)
      .def("set_external_param", &Node::setExternalParam)
      .def("set_external_param_shared_ptr", &Node::setExternalParamSharedPtr)

      // 输入输出边
      .def("set_input", &Node::setInput)
      .def("set_output", &Node::setOutput)
      .def("set_inputs", &Node::setInputs)
      .def("set_outputs", &Node::setOutputs)
      .def("set_input_shared_ptr", &Node::setInputSharedPtr)
      .def("set_output_shared_ptr", &Node::setOutputSharedPtr)
      .def("set_inputs_shared_ptr", &Node::setInputsSharedPtr)
      .def("set_outputs_shared_ptr", &Node::setOutputsSharedPtr)
      .def("get_input", &Node::getInput)
      .def("get_output", &Node::getOutput)
      .def("get_all_input", &Node::getAllInput)
      .def("get_all_output", &Node::getAllOutput)

      // 状态标志
      .def("get_constructed", &Node::getConstructed)
      .def("set_parallel_type", &Node::setParallelType)
      .def("get_parallel_type", &Node::getParallelType)
      .def("set_inner_flag", &Node::setInnerFlag)
      .def("set_initialized_flag", &Node::setInitializedFlag)
      .def("get_initialized", &Node::getInitialized)
      .def("set_time_profile_flag", &Node::setTimeProfileFlag)
      .def("get_time_profile_flag", &Node::getTimeProfileFlag)
      .def("set_debug_flag", &Node::setDebugFlag)
      .def("get_debug_flag", &Node::getDebugFlag)
      .def("set_running_flag", &Node::setRunningFlag)
      .def("is_running", &Node::isRunning)

      // Stream相关
      .def("set_stream", &Node::setStream)
      .def("get_stream", &Node::getStream)

      // 核心功能方法
      .def("init", &Node::init)
      .def("deinit", &Node::deinit)
      .def("get_memory_size", &Node::getMemorySize)
      .def("set_memory", &Node::setMemory)
      .def("updata_input", &Node::updataInput)
      .def("run", &Node::run)
      .def("__call__", &Node::operator())
      .def("functor_without_graph", &Node::functorWithoutGraph)
      .def("functor_with_graph", &Node::functorWithGraph)
      .def("functor_dynamic", &Node::functorDynamic)
      .def("check_inputs", &Node::checkInputs)
      .def("check_outputs", &Node::checkOutputs);
}

}  // namespace dag
}  // namespace nndeploy