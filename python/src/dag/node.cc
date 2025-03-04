#include "nndeploy/dag/node.h"

#include "dag/dag.h"
#include "nndeploy/base/param.h"
#include "nndeploy/dag/edge.h"
#include "nndeploy/dag/graph.h"
#include "nndeploy_api_registry.h"

namespace py = pybind11;
namespace nndeploy {
namespace dag {

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

  py::class_<Node, PyNode<Node>>(m, "Node", py::dynamic_attr())
      .def(py::init<const std::string &>())
      .def(py::init<const std::string &, std::vector<Edge *>,
                    std::vector<Edge *>>())
      .def("get_name", &Node::getName)
      .def("set_graph", &Node::setGraph, py::arg("graph"))
      .def("get_graph", &Node::getGraph, py::return_value_policy::reference)
      .def("set_device_type", &Node::setDeviceType, py::arg("device_type"))
      .def("get_device_type", &Node::getDeviceType)
      .def("set_param", &Node::setParamSharedPtr, py::arg("param"))
      .def("get_param", &Node::getParamSharedPtr)
      .def("set_external_param", &Node::setExternalParam, py::arg("key"),
           py::arg("external_param"))
      .def("get_external_param", &Node::getExternalParam, py::arg("key"))
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
      .def("create_edge", &Node::createEdge, py::arg("name"),
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
      .def("set_compiled_flag", &Node::setCompiledFlag, py::arg("flag"))
      .def("get_compiled_flag", &Node::getCompiledFlag)
      .def("set_graph_flag", &Node::setGraphFlag, py::arg("flag"))
      .def("get_graph_flag", &Node::getGraphFlag)
      .def("set_node_type", &Node::setNodeType, py::arg("node_type"))
      .def("get_node_type", &Node::getNodeType)
      .def("set_stream", &Node::setStream, py::arg("stream"))
      .def("get_stream", &Node::getStream, py::return_value_policy::reference)
      .def("set_input_type_info", 
           [](Node& node, std::shared_ptr<EdgeTypeInfo> input_type_info) {
             return node.setInputTypeInfo(input_type_info);
           }, 
           py::arg("input_type_info"))
      .def("get_input_type_info", &Node::getInputTypeInfo)
      .def("set_output_type_info", 
           [](Node& node, std::shared_ptr<EdgeTypeInfo> output_type_info) {
             return node.setOutputTypeInfo(output_type_info);
           }, 
           py::arg("output_type_info"))
      .def("get_output_type_info", &Node::getOutputTypeInfo)
      .def("init", &Node::init)
      .def("deinit", &Node::deinit)
      .def("get_memory_size", &Node::getMemorySize)
      .def("set_memory", &Node::setMemory, py::arg("buffer"))
      .def("update_input", &Node::updateInput)
      .def("run", &Node::run)
      .def("__call__", &Node::operator(),
           py::arg("inputs"),
           py::arg("outputs_name") = std::vector<std::string>(),
           py::arg("param") = nullptr, py::return_value_policy::reference)
      .def("check_inputs", &Node::checkInputs, py::arg("inputs"))
      .def("check_outputs", 
           py::overload_cast<std::vector<std::string>&>(&Node::checkOutputs),
           py::arg("outputs_name"))
      .def("check_outputs", 
           py::overload_cast<std::vector<Edge*>&>(&Node::checkOutputs),
           py::arg("outputs"))
      .def("is_inputs_changed", &Node::isInputsChanged, py::arg("inputs"))
      .def("get_real_outputs_name", &Node::getRealOutputsName,
           py::arg("outputs_name"));
}

}  // namespace dag
}  // namespace nndeploy