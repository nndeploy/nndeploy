#include "nndeploy/dag/node.h"

#include "nndeploy/base/param.h"
#include "nndeploy/dag/edge.h"
#include "nndeploy/dag/graph.h"
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
    PYBIND11_OVERRIDE(base::Param *, Node, getParam);
  }

  std::shared_ptr<base::Param> getParamSharedPtr() override {
    PYBIND11_OVERRIDE(std::shared_ptr<base::Param>, Node, getParamSharedPtr);
  }

  base::Status setExternalParam(base::Param *external_param) override {
    PYBIND11_OVERRIDE(base::Status, Node, setExternalParam, external_param);
  }

  base::Status setExternalParamSharedPtr(
      std::shared_ptr<base::Param> external_param) override {
    PYBIND11_OVERRIDE(base::Status, Node, setExternalParamSharedPtr,
                      external_param);
  }

  base::Status init() override { PYBIND11_OVERRIDE(base::Status, Node, init); }

  base::Status deinit() override {
    PYBIND11_OVERRIDE(base::Status, Node, deinit);
  }

  int64_t getMemorySize() override {
    PYBIND11_OVERRIDE(int64_t, Node, getMemorySize);
  }

  base::Status setMemory(device::Buffer *buffer) override {
    PYBIND11_OVERRIDE(base::Status, Node, setMemory, buffer);
  }

  base::EdgeUpdateFlag updateInput() override {
    PYBIND11_OVERRIDE(base::EdgeUpdateFlag, Node, updateInput);
  }

  base::Status run() override {
    PYBIND11_OVERRIDE_PURE(base::Status, Node, run);
  }
};

NNDEPLOY_API_PYBIND11_MODULE("dag", m) {
  py::class_<NodeDesc, std::shared_ptr<NodeDesc>>(m, "NodeDesc",
                                                  py::dynamic_attr())
      .def(py::init<const std::string &, std::initializer_list<std::string>,
                    std::initializer_list<std::string>>())
      .def(py::init<const std::string &, std::vector<std::string>,
                    std::vector<std::string>>())
      .def(py::init<const std::string &, const std::string &,
                    std::initializer_list<std::string>,
                    std::initializer_list<std::string>>())
      .def(py::init<const std::string &, const std::string &,
                    std::vector<std::string>, std::vector<std::string>>())
      .def("get_key", &NodeDesc::getKey)
      .def("get_name", &NodeDesc::getName)
      .def("get_inputs", &NodeDesc::getInputs)
      .def("get_outputs", &NodeDesc::getOutputs);

  py::class_<Node, PyNode, std::shared_ptr<Node>>(m, "Node", py::dynamic_attr())
//   py::class_<Node, PyNode>(m, "Node", py::dynamic_attr())
      .def(py::init<const std::string &>())
      .def(py::init<const std::string &, Edge *, Edge *>())
      .def(py::init<const std::string &, std::initializer_list<Edge *>,
                    std::initializer_list<Edge *>>())
      .def(py::init<const std::string &, std::vector<Edge *>,
                    std::vector<Edge *>>())
      .def("get_name", &Node::getName)
      .def("set_graph", &Node::setGraph, py::arg("graph"))
      .def("get_graph", &Node::getGraph, py::return_value_policy::reference)
      .def("set_device_type", &Node::setDeviceType, py::arg("device_type"))
      .def("get_device_type", &Node::getDeviceType)
      .def("set_param_cpp", &Node::setParam, py::arg("param"))
      .def("set_param", &Node::setParamSharedPtr, py::arg("param"))
      .def("get_param_cpp", &Node::getParam, py::return_value_policy::reference)
      .def("get_param", &Node::getParamSharedPtr)
      .def("set_external_param_cpp", &Node::setExternalParam,
           py::arg("external_param"))
      .def("set_external_param", &Node::setExternalParamSharedPtr,
           py::arg("external_param"))
      .def("set_input", &Node::setInput, py::arg("input"),
           py::arg("index") = -1)
      .def("set_output", &Node::setOutput, py::arg("output"),
           py::arg("index") = -1)
      .def("set_inputs", &Node::setInputs, py::arg("inputs"))
      .def("set_outputs", &Node::setOutputs, py::arg("outputs"))
      .def("set_input_shared_ptr", &Node::setInputSharedPtr, py::arg("input"),
           py::arg("index") = -1)
      .def("set_output_shared_ptr", &Node::setOutputSharedPtr,
           py::arg("output"), py::arg("index") = -1)
      .def("set_inputs_shared_ptr", &Node::setInputsSharedPtr,
           py::arg("inputs"))
      .def("set_outputs_shared_ptr", &Node::setOutputsSharedPtr,
           py::arg("outputs"))
      .def("get_input", &Node::getInput, py::arg("index") = 0,
           py::return_value_policy::reference)
      .def("get_output", &Node::getOutput, py::arg("index") = 0,
           py::return_value_policy::reference)
      .def("get_all_input", &Node::getAllInput,
           py::return_value_policy::reference)
      .def("get_all_output", &Node::getAllOutput,
           py::return_value_policy::reference)
      .def("get_constructed", &Node::getConstructed)
      .def("set_parallel_type", &Node::setParallelType,
           py::arg("parallel_type"))
      .def("get_parallel_type", &Node::getParallelType)
      .def("set_inner_flag", &Node::setInnerFlag, py::arg("flag"))
      .def("set_initialized_flag", &Node::setInitializedFlag, py::arg("flag"))
      .def("get_initialized", &Node::getInitialized)
      .def("set_time_profile_flag", &Node::setTimeProfileFlag, py::arg("flag"))
      .def("get_time_profile_flag", &Node::getTimeProfileFlag)
      .def("set_debug_flag", &Node::setDebugFlag, py::arg("flag"))
      .def("get_debug_flag", &Node::getDebugFlag)
      .def("set_running_flag", &Node::setRunningFlag, py::arg("flag"))
      .def("is_running", &Node::isRunning)
      .def("set_stream", &Node::setStream, py::arg("stream"))
      .def("get_stream", &Node::getStream, py::return_value_policy::reference)
      .def("init", &Node::init)
      .def("deinit", &Node::deinit)
      .def("get_memory_size", &Node::getMemorySize)
      .def("set_memory", &Node::setMemory, py::arg("buffer"))
      .def("updata_input", &Node::updateInput)
      .def("run", &Node::run)
      .def("__call__", &Node::operator(), py::arg("inputs"),
           py::arg("outputs_name") = std::vector<std::string>(),
           py::arg("param") = nullptr)
      .def(
          "test",
          [](Node &node, std::vector<Edge *> inputs,
             std::vector<std::string> outputs_name) {
            std::vector<Edge *> cpp_outputs;
            cpp_outputs = node.test(inputs, outputs_name);
            return cpp_outputs;
          },
          py::arg("inputs"),
          py::arg("outputs_name") = std::vector<std::string>(),
          py::return_value_policy::reference)
      .def("functor_without_graph", &Node::functorWithoutGraph,
           py::arg("inputs"),
           py::arg("outputs_name") = std::vector<std::string>(),
           py::arg("param") = nullptr)
      .def("functor_with_graph", &Node::functorWithGraph, py::arg("inputs"),
           py::arg("outputs_name") = std::vector<std::string>(),
           py::arg("param") = nullptr)
      .def("functor_dynamic", &Node::functorDynamic, py::arg("inputs"),
           py::arg("outputs_name") = std::vector<std::string>(),
           py::arg("param") = nullptr)
      .def("check_inputs", &Node::checkInputs, py::arg("inputs"))
      .def("check_outputs", &Node::checkOutputs, py::arg("outputs_name"));
      
}

}  // namespace dag
}  // namespace nndeploy