#include "nndeploy/dag/node.h"

#include "dag/dag.h"
#include "nndeploy/base/param.h"
#include "nndeploy/dag/edge.h"
#include "nndeploy/dag/graph.h"
#include "nndeploy_api_registry.h"

// Windows compatibility: define ssize_t if not available
#ifdef _WIN32
#include <BaseTsd.h>
typedef SSIZE_T ssize_t;
#endif

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
      .def("get_outputs", &NodeDesc::getOutputs)
      .def("serialize",
           py::overload_cast<rapidjson::Value &,
                             rapidjson::Document::AllocatorType &>(
               &NodeDesc::serialize),
           py::arg("json"), py::arg("allocator"))
      .def("serialize", py::overload_cast<>(&NodeDesc::serialize))
      .def("save_file",
           py::overload_cast<const std::string &>(&NodeDesc::saveFile),
           py::arg("path"))
      .def("deserialize",
           py::overload_cast<rapidjson::Value &>(&NodeDesc::deserialize),
           py::arg("json"))
      .def("deserialize",
           py::overload_cast<const std::string &>(&NodeDesc::deserialize),
           py::arg("json_str"))
      .def("load_file",
           py::overload_cast<const std::string &>(&NodeDesc::loadFile),
           py::arg("path"));

  py::class_<Node, PyNode<Node>>(m, "Node", py::dynamic_attr())
      .def(py::init<const std::string &>())
      .def(py::init<const std::string &, std::vector<Edge *>,
                    std::vector<Edge *>>())
      .def("set_key", &Node::setKey, py::arg("key"))
      .def("get_key", &Node::getKey)
      .def("set_name", &Node::setName)
      .def("get_name", &Node::getName)
      .def("set_developer", &Node::setDeveloper, py::arg("developer"))
      .def("get_developer", &Node::getDeveloper)
      .def("set_source", &Node::setSource, py::arg("source"))
      .def("get_source", &Node::getSource)
      .def("set_desc", &Node::setDesc, py::arg("desc"))
      .def("get_desc", &Node::getDesc)
      .def("set_dynamic_input", &Node::setDynamicInput,
           py::arg("is_dynamic_input"))
      .def("set_dynamic_output", &Node::setDynamicOutput,
           py::arg("is_dynamic_output"))
      .def("is_dynamic_input", &Node::isDynamicInput)
      .def("is_dynamic_output", &Node::isDynamicOutput)
      .def("get_input_names", &Node::getInputNames)
      .def("get_output_names", &Node::getOutputNames)
      .def("get_input_name", &Node::getInputName, py::arg("index") = 0)
      .def("get_output_name", &Node::getOutputName, py::arg("index") = 0)
      .def("get_input_index", &Node::getInputIndex, py::arg("name"))
      .def("get_output_index", &Node::getOutputIndex, py::arg("name"))
      .def("get_input_count", &Node::getInputCount)
      .def("get_output_count", &Node::getOutputCount)
      .def("set_input_name", &Node::setInputName, py::arg("name"),
           py::arg("index") = 0)
      .def("set_output_name", &Node::setOutputName, py::arg("name"),
           py::arg("index") = 0)
      .def("set_input_names", &Node::setInputNames, py::arg("names"))
      .def("set_output_names", &Node::setOutputNames, py::arg("names"))
      .def("set_graph", &Node::setGraph, py::arg("graph"))
      .def("get_graph", &Node::getGraph, py::return_value_policy::reference)
      .def("set_device_type", &Node::setDeviceType, py::arg("device_type"))
      .def("get_device_type", &Node::getDeviceType)
      .def("set_param",
           py::overload_cast<std::shared_ptr<base::Param>>(
               &Node::setParamSharedPtr),
           py::arg("param"))
      .def("set_param",
           py::overload_cast<const std::string &, const std::string &>(
               &Node::setParam),
           py::arg("key"), py::arg("value"))
      .def("get_param", &Node::getParamSharedPtr)
      .def("set_external_param", &Node::setExternalParam, py::arg("key"),
           py::arg("external_param"))
      .def("get_external_param", &Node::getExternalParam, py::arg("key"))
      .def("set_version", &Node::setVersion, py::arg("version"))
      .def("get_version", &Node::getVersion)
      .def("set_required_params", &Node::setRequiredParams,
           py::arg("required_params"))
      .def("add_required_param", &Node::addRequiredParam,
           py::arg("required_param"))
      .def("remove_required_param", &Node::removeRequiredParam,
           py::arg("required_param"))
      .def("clear_required_params", &Node::clearRequiredParams)
      .def("get_required_params", &Node::getRequiredParams)
      .def("set_ui_params", &Node::setUiParams, py::arg("ui_params"))
      .def("add_ui_param", &Node::addUiParam, py::arg("ui_param"))
      .def("remove_ui_param", &Node::removeUiParam, py::arg("ui_param"))
      .def("clear_ui_params", &Node::clearUiParams)
      .def("get_ui_params", &Node::getUiParams)
      .def("set_io_params", &Node::setIoParams, py::arg("io_params"))
      .def("add_io_param", &Node::addIoParam, py::arg("io_param"))
      .def("remove_io_param", &Node::removeIoParam, py::arg("io_param"))
      .def("clear_io_params", &Node::clearIoParams)
      .def("get_io_params", &Node::getIoParams)
      .def("set_dropdown_params", &Node::setDropdownParams,
           py::arg("dropdown_params"))
      .def("add_dropdown_param", &Node::addDropdownParam,
           py::arg("dropdown_param"), py::arg("dropdown_values"))
      .def("remove_dropdown_param", &Node::removeDropdownParam,
           py::arg("dropdown_param"))
      .def("clear_dropdown_params", &Node::clearDropdownParams)
      .def("get_dropdown_params", &Node::getDropdownParams)
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
           py::return_value_policy::reference,
           py::call_guard<py::gil_scoped_release>())
      .def("get_output", &Node::getOutput, py::arg("index") = 0,
           py::return_value_policy::reference,
           py::call_guard<py::gil_scoped_release>())
      .def(
          "get_input_data",
          [](Node &node, int index = 0) -> py::object {
            Edge *edge = node.getInput(index);
            if (edge == nullptr) {
              return py::object(py::none());
            }
            // 尝试获取不同类型的数据
            if (auto *tensor = edge->getTensor(&node)) {
              py::gil_scoped_acquire acquire;
              return py::cast(tensor);
            } else if (auto *buffer = edge->getBuffer(&node)) {
              py::gil_scoped_acquire acquire;
              return py::cast(buffer);
            } else if (auto *param = edge->getParam(&node)) {
              py::gil_scoped_acquire acquire;
              return py::cast(param);
            }
#ifdef ENABLE_NNDEPLOY_OPENCV
            else if (auto *mat = edge->getCvMat(&node)) {
              py::gil_scoped_acquire acquire;
              if (mat == nullptr) {
                return py::object(py::array());  // 返回空数组
              }

              std::string format;
              switch (mat->depth()) {
                case CV_8U:
                  format = "B";
                  break;
                case CV_8S:
                  format = "b";
                  break;
                case CV_16U:
                  format = "H";
                  break;
                case CV_16S:
                  format = "h";
                  break;
                case CV_32S:
                  format = "i";
                  break;
                case CV_32F:
                  format = "f";
                  break;
                case CV_64F:
                  format = "d";
                  break;
                default:
                  throw std::runtime_error("Unsupported cv::Mat data type");
              }

              std::vector<ssize_t> shape;
              std::vector<ssize_t> strides;

              if (mat->channels() == 1) {
                // 单通道图像
                shape = {static_cast<ssize_t>(mat->rows),
                         static_cast<ssize_t>(mat->cols)};
                strides = {static_cast<ssize_t>(mat->step[0]),
                           static_cast<ssize_t>(mat->elemSize1())};
              } else {
                // 多通道图像
                shape = {static_cast<ssize_t>(mat->rows),
                         static_cast<ssize_t>(mat->cols),
                         static_cast<ssize_t>(mat->channels())};
                strides = {
                    static_cast<ssize_t>(mat->step[0]),
                    static_cast<ssize_t>(mat->elemSize1() * mat->channels()),
                    static_cast<ssize_t>(mat->elemSize1())};
              }

              return py::object(
                  py::array(py::buffer_info(mat->data,         // 数据指针
                                            mat->elemSize1(),  // 单个元素大小
                                            format,            // 数据格式
                                            shape.size(),      // 维度数
                                            shape,             // 形状
                                            strides            // 步长
                                            )));
            }
#endif
            else if (auto *obj = edge->get<PyObject>(&node)) {
              if (!obj) {
                return py::object(py::none());
              }
              return py::reinterpret_borrow<py::object>(obj);
            }
            return py::object(py::none());
          },
          py::arg("index") = 0, py::return_value_policy::reference,
          py::call_guard<py::gil_scoped_release>())
      .def(
          "set_output_data",
          [](Node &node, py::object obj, int index = 0,
             bool is_external = true) -> base::Status {
            Edge *edge = node.getOutput(index);
            if (edge == nullptr) {
              return base::kStatusCodeErrorNullParam;
            }
            auto *wrapper = new PyObjectWrapper(obj.ptr());
            // 检查输入对象类型
            if (py::isinstance<py::array>(obj)) {
              // numpy array类型
              auto buffer = obj.cast<py::array>();
              //        // 获取numpy数组的维度和形状
              py::buffer_info info = buffer.request();
              char kind = info.format.front();
              int channels = (info.ndim == 3) ? info.shape[2] : 1;
              int cv_depth;
#ifdef ENABLE_NNDEPLOY_OPENCV
              // 根据numpy数组的数据类型和每个元素的大小来确定OpenCV的数据类型
              switch (kind) {
                case 'B':
                  cv_depth = CV_8U;
                  break;
                case 'b':
                  cv_depth = CV_8S;
                  break;
                case 'H':
                  cv_depth = CV_16U;
                  break;
                case 'h':
                  cv_depth = CV_16S;
                  break;
                case 'i':
                  cv_depth = CV_32S;
                  break;
                case 'f':
                  cv_depth = CV_32F;
                  break;
                case 'd':
                  cv_depth = CV_64F;
                  break;
                default:
                  throw std::runtime_error("Unsupported data type kind: " +
                                           std::string(1, kind));
              }
              int type = CV_MAKETYPE(cv_depth, channels);
              cv::Mat *mat =
                  new cv::Mat(info.shape[0], info.shape[1], type, info.ptr);
              base::Status status = edge->set4py(wrapper, mat, false);
              if (status != base::StatusCode::kStatusCodeOk) {
                throw std::runtime_error("Failed to set cv::Mat");
              }
              return status;
#else
              NNDEPLOY_LOGE("set cv::Mat is not supported");
              base::Status status =
                  base::StatusCode::kStatusCodeErrorNotSupport;
              return status;
#endif
            } else if (py::isinstance<device::Tensor>(obj)) {
              // Tensor类型
              auto *tensor = obj.cast<device::Tensor *>();
              base::Status status = edge->set4py(wrapper, tensor);
              if (status != base::StatusCode::kStatusCodeOk) {
                throw std::runtime_error("Failed to set device::Tensor");
              }
              return status;
            } else if (py::isinstance<device::Buffer>(obj)) {
              // Buffer类型
              auto *buffer = obj.cast<device::Buffer *>();
              base::Status status = edge->set4py(wrapper, buffer);
              if (status != base::StatusCode::kStatusCodeOk) {
                throw std::runtime_error("Failed to set device::Buffer");
              }
              return status;
            } else if (py::isinstance<base::Param>(obj)) {
              // Param类型
              auto param = obj.cast<std::shared_ptr<base::Param>>();
              base::Status status = edge->set4py(wrapper, param.get());
              if (status != base::StatusCode::kStatusCodeOk) {
                throw std::runtime_error("Failed to set base::Param");
              }
              return status;
            } else {
              base::Status status = edge->set4py(wrapper, obj.ptr());
              if (status != base::StatusCode::kStatusCodeOk) {
                throw std::runtime_error("Failed to set base::Param");
              }
              return status;
            }
          },
          py::arg("obj"), py::arg("index") = 0, py::arg("is_external") = true)
      .def("get_all_input", &Node::getAllInput,
           py::return_value_policy::reference,
           py::call_guard<py::gil_scoped_release>())
      .def("get_all_output", &Node::getAllOutput,
           py::return_value_policy::reference,
           py::call_guard<py::gil_scoped_release>())
      .def("create_internal_output_edge", &Node::createInternalOutputEdge,
           py::arg("name"), py::return_value_policy::reference)
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
      .def("set_running_flag", &Node::setRunningFlag, py::arg("flag"),
           py::call_guard<py::gil_scoped_release>())
      .def("is_running", &Node::isRunning)
      .def("get_run_size", &Node::getRunSize)
      .def("get_completed_size", &Node::getCompletedSize)
      .def("get_run_status", &Node::getRunStatus,
           py::call_guard<py::gil_scoped_release>())
      .def("set_trace_flag", &Node::setTraceFlag, py::arg("flag"))
      .def("get_trace_flag", &Node::getTraceFlag)
      .def("set_graph_flag", &Node::setGraphFlag, py::arg("flag"))
      .def("get_graph_flag", &Node::getGraphFlag)
      .def("set_node_type", &Node::setNodeType, py::arg("node_type"))
      .def("get_node_type", &Node::getNodeType)
      .def("set_io_type", &Node::setIoType, py::arg("io_type"))
      .def("get_io_type", &Node::getIoType)
      .def("set_loop_count", &Node::setLoopCount, py::arg("loop_count"))
      .def("get_loop_count", &Node::getLoopCount)
      .def("set_stream", &Node::setStream, py::arg("stream"))
      .def("get_stream", &Node::getStream, py::return_value_policy::reference)
      .def(
          "set_input_type_info",
          [](Node &node, std::shared_ptr<EdgeTypeInfo> input_type_info,
             std::string desc = "") {
            return node.setInputTypeInfo(input_type_info, desc);
          },
          py::arg("input_type_info"), py::arg("desc") = "")
      .def("get_input_type_info", &Node::getInputTypeInfo)
      .def(
          "set_output_type_info",
          [](Node &node, std::shared_ptr<EdgeTypeInfo> output_type_info,
             std::string desc = "") {
            return node.setOutputTypeInfo(output_type_info, desc);
          },
          py::arg("output_type_info"), py::arg("desc") = "")
      .def("get_output_type_info", &Node::getOutputTypeInfo)
      .def("default_param", &Node::defaultParam)
      .def("init", &Node::init, py::call_guard<py::gil_scoped_release>())
      .def("deinit", &Node::deinit, py::call_guard<py::gil_scoped_release>())
      .def("get_memory_size", &Node::getMemorySize)
      .def("set_memory", &Node::setMemory, py::arg("buffer"))
      .def("update_input", &Node::updateInput,
           py::call_guard<py::gil_scoped_release>())
      .def("run", &Node::run, py::call_guard<py::gil_scoped_release>())
      .def("synchronize", &Node::synchronize,
           py::call_guard<py::gil_scoped_release>())
      .def("forward", py::overload_cast<std::vector<Edge *>>(&Node::forward),
           py::arg("inputs"), py::return_value_policy::reference,
           py::call_guard<py::gil_scoped_release>())
      .def("forward", py::overload_cast<>(&Node::forward),
           py::return_value_policy::reference,
           py::call_guard<py::gil_scoped_release>())
      .def("forward", py::overload_cast<Edge *>(&Node::forward),
           py::arg("input"), py::return_value_policy::reference,
           py::call_guard<py::gil_scoped_release>())
      .def("__call__",
           py::overload_cast<std::vector<Edge *>>(&Node::operator()),
           py::arg("inputs"), py::return_value_policy::reference,
           py::call_guard<py::gil_scoped_release>())
      .def("__call__", py::overload_cast<>(&Node::operator()),
           py::return_value_policy::reference,
           py::call_guard<py::gil_scoped_release>())
      .def("__call__", py::overload_cast<Edge *>(&Node::operator()),
           py::arg("input"), py::return_value_policy::reference,
           py::call_guard<py::gil_scoped_release>())
      .def("check_inputs", &Node::checkInputs, py::arg("inputs"))
      .def("check_outputs",
           py::overload_cast<std::vector<std::string> &>(&Node::checkOutputs),
           py::arg("outputs_name"))
      .def("check_outputs",
           py::overload_cast<std::vector<Edge *> &>(&Node::checkOutputs),
           py::arg("outputs"))
      .def("is_inputs_changed", &Node::isInputsChanged, py::arg("inputs"))
      .def("to_static_graph", &Node::toStaticGraph,
           py::return_value_policy::reference)
      .def("get_real_outputs_name", &Node::getRealOutputsName)
      .def("serialize",
           py::overload_cast<rapidjson::Value &,
                             rapidjson::Document::AllocatorType &>(
               &Node::serialize),
           py::arg("json"), py::arg("allocator"))
      .def("serialize", py::overload_cast<>(&Node::serialize))
      .def("save_file", py::overload_cast<const std::string &>(&Node::saveFile),
           py::arg("path"))
      .def("deserialize",
           py::overload_cast<rapidjson::Value &>(&Node::deserialize),
           py::arg("json"))
      .def("deserialize",
           py::overload_cast<const std::string &>(&Node::deserialize),
           py::arg("json_str"))
      .def("load_file", py::overload_cast<const std::string &>(&Node::loadFile),
           py::arg("path"));

  py::class_<ConstNode, Node, PyConstNode<ConstNode>>(m, "ConstNode",
                                                      py::dynamic_attr())
      .def(py::init<const std::string &>())
      .def(py::init<const std::string &, std::vector<Edge *>,
                    std::vector<Edge *>>())
      .def("update_input", &ConstNode::updateInput)
      .def("init", &ConstNode::init)
      .def("deinit", &ConstNode::deinit)
      .def("run", &ConstNode::run);

  py::class_<CompositeNode, Node, PyCompositeNode<CompositeNode>>(
      m, "CompositeNode", py::dynamic_attr())
      .def(py::init<const std::string &>())
      .def(py::init<const std::string &, std::vector<Edge *>,
                    std::vector<Edge *>>())
      .def("set_input", &CompositeNode::setInput, py::arg("input"),
           py::arg("index") = -1)
      .def("set_output", &CompositeNode::setOutput, py::arg("output"),
           py::arg("index") = -1)
      .def("set_inputs", &CompositeNode::setInputs, py::arg("inputs"))
      .def("set_outputs", &CompositeNode::setOutputs, py::arg("outputs"))
      .def("create_edge", &CompositeNode::createEdge, py::arg("name"),
           py::return_value_policy::reference)
      .def("add_edge", &CompositeNode::addEdge, py::arg("edge"),
           py::arg("is_external") = true)
      .def("get_edge", &CompositeNode::getEdge, py::arg("name"),
           py::return_value_policy::reference)
      .def("update_edge", &CompositeNode::updteEdge, py::arg("edge_wrapper"),
           py::arg("edge"), py::arg("is_external") = true)
      .def("create_node",
           py::overload_cast<const NodeDesc &>(&CompositeNode::createNode4Py),
           py::arg("desc"), py::return_value_policy::take_ownership)
      .def("set_node_desc", &CompositeNode::setNodeDesc, py::arg("node"),
           py::arg("desc"))
      .def(
          "add_node",
          [](CompositeNode &c, Node *node) {
            bool is_external = true;
            c.addNode(node, is_external);
          },
          py::arg("node"), py::keep_alive<1, 2>())
      .def("update_node_io", &CompositeNode::updateNodeIO, py::arg("node"),
           py::arg("inputs"), py::arg("outputs"))
      .def("mark_input_edge", &CompositeNode::markInputEdge, py::arg("inputs"))
      .def("mark_output_edge", &CompositeNode::markOutputEdge,
           py::arg("outputs"))
      .def("get_node", &CompositeNode::getNode, py::arg("name"),
           py::return_value_policy::reference)
      .def("get_node_by_key", &CompositeNode::getNodeByKey, py::arg("key"),
           py::return_value_policy::reference)
      .def("get_nodes_by_key", &CompositeNode::getNodesByKey, py::arg("key"),
           py::return_value_policy::reference)
      .def("set_node_param", &CompositeNode::setNodeParamSharedPtr,
           py::arg("node_name"), py::arg("param"))
      .def("get_node_param", &CompositeNode::getNodeParamSharedPtr,
           py::arg("node_name"), py::return_value_policy::reference)
      .def("default_param", &CompositeNode::defaultParam)
      .def("init", &CompositeNode::init)
      .def("deinit", &CompositeNode::deinit)
      .def("run", &CompositeNode::run)
      .def("get_edge_wrapper",
           py::overload_cast<Edge *>(&CompositeNode::getEdgeWrapper),
           py::arg("edge"), py::return_value_policy::reference)
      .def("get_edge_wrapper",
           py::overload_cast<const std::string &>(
               &CompositeNode::getEdgeWrapper),
           py::arg("name"), py::return_value_policy::reference)
      .def("get_node_wrapper",
           py::overload_cast<Node *>(&CompositeNode::getNodeWrapper),
           py::arg("node"), py::return_value_policy::reference)
      .def("get_node_wrapper",
           py::overload_cast<const std::string &>(
               &CompositeNode::getNodeWrapper),
           py::arg("name"), py::return_value_policy::reference)
      .def("serialize",
           py::overload_cast<rapidjson::Value &,
                             rapidjson::Document::AllocatorType &>(
               &CompositeNode::serialize),
           py::arg("json"), py::arg("allocator"))
      .def("serialize", py::overload_cast<>(&CompositeNode::serialize))
      .def("deserialize",
           py::overload_cast<rapidjson::Value &>(&CompositeNode::deserialize),
           py::arg("json"))
      .def("deserialize",
           py::overload_cast<const std::string &>(&CompositeNode::deserialize),
           py::arg("json_str"))
      .def("sort_dfs", &CompositeNode::sortDFS);

  py::class_<NodeCreator, PyNodeCreator<NodeCreator>,
             std::shared_ptr<NodeCreator>>(m, "NodeCreator")
      .def(py::init<>())
      .def("create_node", &NodeCreator::createNode,
           py::return_value_policy::take_ownership);

  m.def("get_node_keys", &getNodeKeys);

  m.def("register_node",
        [](const std::string &node_key, std::shared_ptr<NodeCreator> creator) {
          NodeFactory *instance = getGlobalNodeFactory();
          if (instance != nullptr) {
            instance->registerNode(node_key, creator);
            // NNDEPLOY_LOGI("register node success: %s\n", node_key.c_str());
          } else {
            NNDEPLOY_LOGE("register node failed: %s\n", node_key.c_str());
          }
        });

  m.def(
      "create_node",
      py::overload_cast<const std::string &, const std::string &>(&createNode),
      py::arg("node_key"), py::arg("node_name"),
      py::return_value_policy::take_ownership);

  m.def("create_node",
        py::overload_cast<const std::string &, const std::string &,
                          std::initializer_list<Edge *>,
                          std::initializer_list<Edge *>>(&createNode),
        py::arg("node_key"), py::arg("node_name"), py::arg("inputs"),
        py::arg("outputs"), py::return_value_policy::take_ownership);

  m.def(
      "create_node",
      py::overload_cast<const std::string &, const std::string &,
                        std::vector<Edge *>, std::vector<Edge *>>(&createNode),
      py::arg("node_key"), py::arg("node_name"), py::arg("inputs"),
      py::arg("outputs"), py::return_value_policy::take_ownership);

  // 定义Graph类
  py::class_<Graph, Node, PyGraph<Graph>>(m, "Graph", py::dynamic_attr())
      .def(py::init<const std::string &>())
      .def(py::init<const std::string &, std::vector<Edge *>,
                    std::vector<Edge *>>())
      .def("add_image_url", &Graph::addImageUrl, py::arg("url"))
      .def("remove_image_url", &Graph::removeImageUrl, py::arg("url"))
      .def("add_video_url", &Graph::addVideoUrl, py::arg("url"))
      .def("remove_video_url", &Graph::removeVideoUrl, py::arg("url"))
      .def("add_audio_url", &Graph::addAudioUrl, py::arg("url"))
      .def("remove_audio_url", &Graph::removeAudioUrl, py::arg("url"))
      .def("add_model_url", &Graph::addModelUrl, py::arg("url"))
      .def("remove_model_url", &Graph::removeModelUrl, py::arg("url"))
      .def("add_other_url", &Graph::addOtherUrl, py::arg("url"))
      .def("remove_other_url", &Graph::removeOtherUrl, py::arg("url"))
      .def("get_image_url", &Graph::getImageUrl)
      .def("get_video_url", &Graph::getVideoUrl)
      .def("get_audio_url", &Graph::getAudioUrl)
      .def("get_model_url", &Graph::getModelUrl)
      .def("get_other_url", &Graph::getOtherUrl)
      .def("set_edge_queue_max_size", &Graph::setEdgeQueueMaxSize,
           py::arg("queue_max_size"))
      .def("get_edge_queue_max_size", &Graph::getEdgeQueueMaxSize)
      .def("set_edge_queue_overflow_policy",
           &Graph::setEdgeQueueOverflowPolicy, py::arg("policy"),
           py::arg("drop_count") = 1)
      .def("get_edge_queue_overflow_policy",
           &Graph::getEdgeQueueOverflowPolicy)
      .def("get_edge_queue_drop_count", &Graph::getEdgeQueueDropCount)
      .def("set_input", &Graph::setInput, py::arg("input"),
           py::arg("index") = -1)
      .def("set_output", &Graph::setOutput, py::arg("output"),
           py::arg("index") = -1)
      .def("set_inputs", &Graph::setInputs, py::arg("inputs"))
      .def("set_outputs", &Graph::setOutputs, py::arg("outputs"))
      //  .def("set_input_shared_ptr", &Graph::setInputSharedPtr,
      //  py::arg("input"),
      //       py::arg("index") = -1)
      //  .def("set_output_shared_ptr", &Graph::setOutputSharedPtr,
      //       py::arg("output"), py::arg("index") = -1)
      //  .def("set_inputs_shared_ptr", &Graph::setInputsSharedPtr,
      //       py::arg("inputs"))
      //  .def("set_outputs_shared_ptr", &Graph::setOutputsSharedPtr,
      //       py::arg("outputs"))
      .def("create_edge", &Graph::createEdge, py::arg("name"),
           py::return_value_policy::reference)
      .def(
          "add_edge",
          [](Graph &g, Edge *edge) {
            bool is_external = true;
            g.addEdge(edge, is_external);
          },
          py::arg("edge"), py::keep_alive<1, 2>())
      .def("update_edge", &Graph::updteEdge, py::arg("edge_wrapper"),
           py::arg("edge"), py::arg("is_external") = true)
      .def("get_edge", &Graph::getEdge, py::arg("name"),
           py::return_value_policy::reference)
      .def("create_node",
           py::overload_cast<const std::string &, const std::string &>(
               &Graph::createNode4Py),
           py::arg("key"), py::arg("name"),
           py::return_value_policy::take_ownership)
      .def("create_node",
           py::overload_cast<const NodeDesc &>(&Graph::createNode4Py),
           py::arg("desc"), py::return_value_policy::take_ownership)
      .def(
          "set_node_desc",
          [](Graph &g, Node *node, const NodeDesc &desc) {
            //   NNDEPLOY_LOGE("set_node_desc[%s, %p] success!\n",
            //                 node->getName().c_str(), node);
            return g.setNodeDesc(node, desc);
          },
          py::arg("node"), py::arg("desc"))
      .def(
          "add_node",
          [](Graph &g, Node *node) {
            bool is_external = true;
            g.addNode(node, is_external);
          },
          py::keep_alive<1, 2>(), py::arg("node"))
      .def("get_node", py::overload_cast<const std::string &>(&Graph::getNode),
           py::arg("name"), py::return_value_policy::reference)
      .def("get_node", py::overload_cast<int>(&Graph::getNode),
           py::arg("index"), py::return_value_policy::reference)
      //  .def("get_node_shared_ptr", &Graph::getNodeSharedPtr, py::arg("name"))
      .def("get_node_by_key", &Graph::getNodeByKey, py::arg("key"),
           py::return_value_policy::reference)
      .def("get_nodes_by_key", &Graph::getNodesByKey, py::arg("key"),
           py::return_value_policy::reference)
      .def("get_node_count", &Graph::getNodeCount)
      .def("get_nodes", &Graph::getNodes, py::return_value_policy::reference)
      .def("get_nodes_recursive", &Graph::getNodesRecursive,
           py::return_value_policy::reference)
      .def("get_nodes_name", &Graph::getNodesName,
           py::return_value_policy::reference)
      .def("get_nodes_name_recursive", &Graph::getNodesNameRecursive,
           py::return_value_policy::reference)
      .def("get_input_node", &Graph::getInputNode, py::arg("index"),
           py::return_value_policy::reference)
      .def("get_output_node", &Graph::getOutputNode, py::arg("index"),
           py::return_value_policy::reference)
      .def("get_infer_node", &Graph::getInferNode, py::arg("index"),
           py::return_value_policy::reference)
      .def("get_nodes_run_status", &Graph::getNodesRunStatus,
           py::call_guard<py::gil_scoped_release>())
      .def("get_nodes_run_status_recursive", &Graph::getNodesRunStatusRecursive,
           py::call_guard<py::gil_scoped_release>())
      .def("set_node_param", &Graph::setNodeParamSharedPtr,
           py::arg("node_name"), py::arg("param"))
      .def("get_node_param", &Graph::getNodeParamSharedPtr,
           py::arg("node_name"))
      .def("set_external_param", &Graph::setExternalParam, py::arg("key"),
           py::arg("param"))
      .def("get_external_param", &Graph::getExternalParam, py::arg("key"))
      .def("set_node_parallel_type", &Graph::setNodeParallelType,
           py::arg("node_name"), py::arg("parallel_type"))
      .def("set_graph_node_share_stream", &Graph::setGraphNodeShareStream,
           py::arg("flag"))
      .def("get_graph_node_share_stream", &Graph::getGraphNodeShareStream)
      .def("set_loop_count", &Graph::setLoopCount, py::arg("loop_count"))
      .def("get_loop_count", &Graph::getLoopCount)
      .def("get_loop_count_map", &Graph::getLoopCountMap)
      .def("update_node_io", &Graph::updateNodeIO, py::arg("node"),
           py::arg("inputs"), py::arg("outputs"))
      .def("mark_input_edge", &Graph::markInputEdge, py::arg("inputs"))
      .def("mark_output_edge", &Graph::markOutputEdge, py::arg("outputs"))
      .def("default_param", &Graph::defaultParam)
      .def("init", &Graph::init, py::call_guard<py::gil_scoped_release>())
      .def(
          "deinit",
          [](Graph &g) {
            //   NNDEPLOY_LOGE("deinit start!\n");
            return g.deinit();
          },
          py::call_guard<py::gil_scoped_release>())
      .def("run", &Graph::run, py::call_guard<py::gil_scoped_release>())
      .def("synchronize", &Graph::synchronize,
           py::call_guard<py::gil_scoped_release>())
      .def("interrupt", &Graph::interrupt,
           py::call_guard<py::gil_scoped_release>())
      .def("forward", py::overload_cast<std::vector<Edge *>>(&Graph::forward),
           py::arg("inputs"), py::keep_alive<1, 2>(),
           py::return_value_policy::reference,
           py::call_guard<py::gil_scoped_release>())
      .def("forward", py::overload_cast<>(&Graph::forward),
           py::return_value_policy::reference,
           py::call_guard<py::gil_scoped_release>())
      .def("forward", py::overload_cast<Edge *>(&Graph::forward),
           py::arg("input"), py::keep_alive<1, 2>(),
           py::return_value_policy::reference,
           py::call_guard<py::gil_scoped_release>())
      .def("__call__",
           py::overload_cast<std::vector<Edge *>>(&Graph::operator()),
           py::arg("inputs"), py::keep_alive<1, 2>(),
           py::return_value_policy::reference,
           py::call_guard<py::gil_scoped_release>())
      .def("__call__", py::overload_cast<>(&Graph::operator()),
           py::return_value_policy::reference,
           py::call_guard<py::gil_scoped_release>())
      .def("__call__", py::overload_cast<Edge *>(&Graph::operator()),
           py::arg("input"), py::keep_alive<1, 2>(),
           py::return_value_policy::reference,
           py::call_guard<py::gil_scoped_release>())
      .def("dump", [](Graph &g) { g.dump(std::cout); })
      .def("set_trace_flag", &Graph::setTraceFlag, py::arg("flag"))
      // 绑定Graph类的trace方法到Python
      // 参数:
      //   inputs: 输入的边列表
      // 返回:
      //   返回追踪后的边列表
      // 说明:
      //   1. py::keep_alive<1,2>() 确保Graph对象在inputs存在期间不会被销毁
      //   2. py::return_value_policy::reference 返回引用,由Python管理内存
      //   3. 该方法用于追踪图的执行流程,帮助调试和性能分析
      .def("trace", py::overload_cast<std::vector<Edge *>>(&Graph::trace),
           py::arg("inputs"), py::keep_alive<1, 2>(),
           py::return_value_policy::reference)
      .def("trace", py::overload_cast<>(&Graph::trace),
           py::return_value_policy::reference)
      .def("trace", py::overload_cast<Edge *>(&Graph::trace), py::arg("input"),
           py::keep_alive<1, 2>(), py::return_value_policy::reference)
      .def("to_static_graph", &Graph::toStaticGraph,
           py::return_value_policy::reference)
      .def("get_edge_wrapper",
           py::overload_cast<Edge *>(&Graph::getEdgeWrapper), py::arg("edge"),
           py::return_value_policy::reference)
      .def("get_edge_wrapper",
           py::overload_cast<const std::string &>(&Graph::getEdgeWrapper),
           py::arg("name"), py::return_value_policy::reference)
      .def("get_node_wrapper",
           py::overload_cast<Node *>(&Graph::getNodeWrapper), py::arg("node"),
           py::return_value_policy::reference)
      .def("get_node_wrapper",
           py::overload_cast<const std::string &>(&Graph::getNodeWrapper),
           py::arg("name"), py::return_value_policy::reference)
      .def("serialize",
           py::overload_cast<rapidjson::Value &,
                             rapidjson::Document::AllocatorType &>(
               &Graph::serialize),
           py::arg("json"), py::arg("allocator"))
      .def("serialize", py::overload_cast<>(&Graph::serialize))
      .def("deserialize",
           py::overload_cast<rapidjson::Value &>(&Graph::deserialize),
           py::arg("json"))
      .def("deserialize",
           py::overload_cast<const std::string &>(&Graph::deserialize),
           py::arg("json_str"))
      .def("set_unused_node_names",
           py::overload_cast<const std::string &>(&Graph::setUnusedNodeNames),
           py::arg("node_name"))
      .def("set_unused_node_names",
           py::overload_cast<const std::set<std::string> &>(
               &Graph::setUnusedNodeNames),
           py::arg("node_names"))
      .def(
          "remove_unused_node_names",
          py::overload_cast<const std::string &>(&Graph::removeUnusedNodeNames),
          py::arg("node_name"))
      .def("remove_unused_node_names",
           py::overload_cast<const std::set<std::string> &>(
               &Graph::removeUnusedNodeNames),
           py::arg("node_names"))
      .def("get_unused_node_names", &Graph::getUnusedNodeNames)
      .def("set_node_value",
           py::overload_cast<const std::string &>(&Graph::setNodeValue),
           py::arg("node_value_str"))
      .def("set_node_value",
           py::overload_cast<const std::string &, const std::string &,
                             const std::string &>(&Graph::setNodeValue),
           py::arg("node_name"), py::arg("key"), py::arg("value"))
      .def("set_node_value",
           py::overload_cast<
               std::map<std::string, std::map<std::string, std::string>>>(
               &Graph::setNodeValue),
           py::arg("node_value_map"))
      .def("get_node_value", &Graph::getNodeValue);

  //   m.def("serialize", py::overload_cast<Graph *>(&serialize),
  //   py::arg("graph")); m.def("save_file", &saveFile, py::arg("graph"),
  //   py::arg("path")); m.def("deserialize", py::overload_cast<const
  //   std::string &>(&deserialize),
  //         py::arg("json_str"), py::return_value_policy::take_ownership);
  //   m.def("load_file", &loadFile, py::arg("path"),
  //         py::return_value_policy::take_ownership);

  py::class_<Loop, Graph, PyLoop<Loop>>(m, "Loop", py::dynamic_attr())
      .def(py::init<const std::string &>())
      .def(py::init<const std::string &, std::vector<Edge *>,
                    std::vector<Edge *>>())
      .def("init", &Loop::init)
      .def("deinit", &Loop::deinit)
      .def("loops", &Loop::loops)
      .def("run", &Loop::run);

  py::class_<Condition, Graph, PyCondition<Condition>>(m, "Condition",
                                                       py::dynamic_attr())
      .def(py::init<const std::string &>())
      .def(py::init<const std::string &, std::vector<Edge *>,
                    std::vector<Edge *>>())
      .def("init", &Condition::init)
      .def("deinit", &Condition::deinit)
      .def("choose", &Condition::choose)
      .def("run", &Condition::run);

  py::class_<RunningCondition, Condition, PyRunningCondition<RunningCondition>>(
      m, "RunningCondition", py::dynamic_attr())
      .def(py::init<const std::string &>())
      .def(py::init<const std::string &, std::vector<Edge *>,
                    std::vector<Edge *>>())
      .def("choose", &RunningCondition::choose);
}

}  // namespace dag
}  // namespace nndeploy