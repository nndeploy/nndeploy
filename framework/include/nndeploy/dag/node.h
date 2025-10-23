
#ifndef _NNDEPLOY_DAG_NODE_H_
#define _NNDEPLOY_DAG_NODE_H_

#include "nndeploy/base/any.h"
#include "nndeploy/base/common.h"
#include "nndeploy/base/glic_stl_include.h"
#include "nndeploy/base/log.h"
#include "nndeploy/base/macro.h"
#include "nndeploy/base/object.h"
#include "nndeploy/base/status.h"
#include "nndeploy/base/string.h"
#include "nndeploy/base/time_profiler.h"
#include "nndeploy/dag/base.h"
#include "nndeploy/dag/edge.h"
#include "nndeploy/device/buffer.h"
#include "nndeploy/device/device.h"
#include "nndeploy/device/memory_pool.h"
#include "nndeploy/device/tensor.h"
#include "rapidjson/document.h"
#include "rapidjson/stringbuffer.h"
#include "rapidjson/writer.h"

namespace nndeploy {
namespace dag {

// Forward declarations
class Node;
class Graph;
class CompositeNode;

/**
 * @brief Node description class
 * @details Used to describe basic information of a node, including node name, input/output edge names, etc.
 */
class NNDEPLOY_CC_API NodeDesc {
 public:
  NodeDesc() = default;
  
  /**
   * @brief Constructor
   * @param node_name Node name
   * @param inputs Input edge name list
   * @param outputs Output edge name list
   */
  NodeDesc(const std::string &node_name,
           std::initializer_list<std::string> inputs,
           std::initializer_list<std::string> outputs)
      : node_name_(node_name), inputs_(inputs), outputs_(outputs) {}
      
  /**
   * @brief Constructor
   * @param node_name Node name
   * @param inputs Input edge name vector
   * @param outputs Output edge name vector
   */
  NodeDesc(const std::string &node_name, std::vector<std::string> inputs,
           std::vector<std::string> outputs)
      : node_name_(node_name), inputs_(inputs), outputs_(outputs) {}
      
  /**
   * @brief Constructor
   * @param node_key Node key
   * @param node_name Node name
   * @param inputs Input edge name list
   * @param outputs Output edge name list
   */
  NodeDesc(const std::string &node_key, const std::string &node_name,
           std::initializer_list<std::string> inputs,
           std::initializer_list<std::string> outputs)
      : node_key_(node_key),
        node_name_(node_name),
        inputs_(inputs),
        outputs_(outputs) {}
        
  /**
   * @brief Constructor
   * @param node_key Node key
   * @param node_name Node name
   * @param inputs Input edge name vector
   * @param outputs Output edge name vector
   */
  NodeDesc(const std::string &node_key, const std::string &node_name,
           std::vector<std::string> inputs, std::vector<std::string> outputs)
      : node_key_(node_key),
        node_name_(node_name),
        inputs_(inputs),
        outputs_(outputs) {}

  virtual ~NodeDesc() = default;

  /**
   * @brief Get node key
   * @return Node key
   */
  std::string getKey() const { return node_key_; }

  /**
   * @brief Get node name
   * @return Node name
   */
  std::string getName() const { return node_name_; }

  /**
   * @brief Get input edge name list
   * @return Input edge name vector
   */
  std::vector<std::string> getInputs() const { return inputs_; }

  /**
   * @brief Get output edge name list
   * @return Output edge name vector
   */
  std::vector<std::string> getOutputs() const { return outputs_; }

  // JSON serialization related methods
  /**
   * @brief Serialize to JSON
   * @param json JSON value object
   * @param allocator JSON allocator
   * @return Operation status
   */
  base::Status serialize(rapidjson::Value &json,
                         rapidjson::Document::AllocatorType &allocator);
                         
  /**
   * @brief Serialize to JSON string
   * @return JSON string
   */
  std::string serialize();
  
  /**
   * @brief Save to file
   * @param path File path
   * @return Operation status
   */
  base::Status saveFile(const std::string &path);
  
  /**
   * @brief Deserialize from JSON
   * @param json JSON value object
   * @return Operation status
   */
  base::Status deserialize(rapidjson::Value &json);
  
  /**
   * @brief Deserialize from JSON string
   * @param json_str JSON string
   * @return Operation status
   */
  base::Status deserialize(const std::string &json_str);
  
  /**
   * @brief Load from file
   * @param path File path
   * @return Operation status
   */
  base::Status loadFile(const std::string &path);

 private:
  std::string node_key_;    ///< Node key
  std::string node_name_;   ///< Node name
  std::vector<std::string> inputs_;   ///< Input edge name list
  std::vector<std::string> outputs_;  ///< Output edge name list
};

/**
 * @brief Node base class
 * @details Basic computational unit in DAG graph, each node is responsible for allocating memory for its output edges
 * @note Each node is responsible for allocating memory for its output edges
 */
class NNDEPLOY_CC_API Node {
 public:
  /**
   * @brief Constructor
   * @param name Node name
   */
  Node(const std::string &name);
  
  /**
   * @brief Constructor
   * @param name Node name
   * @param inputs Input edge list
   * @param outputs Output edge list
   */
  Node(const std::string &name, std::vector<Edge *> inputs,
       std::vector<Edge *> outputs);

  virtual ~Node();

  // Basic property setting and getting methods
  /**
   * @brief Set node key
   * @param key Node key
   */
  void setKey(const std::string &key);
  
  /**
   * @brief Get node key
   * @return Node key
   */
  std::string getKey();
  
  /**
   * @brief Set node name
   * @param name Node name
   */
  void setName(const std::string &name);
  
  /**
   * @brief Get node name
   * @return Node name
   */
  std::string getName();
  
  /**
   * @brief Set developer information
   * @param developer Developer name
   */
  void setDeveloper(const std::string &developer);
  
  /**
   * @brief Get developer information
   * @return Developer name
   */
  std::string getDeveloper();
  
  /**
   * @brief Set source information
   * @param source Source information
   */
  void setSource(const std::string &source);
  
  /**
   * @brief Get source information
   * @return Source information
   */
  std::string getSource();
  
  /**
   * @brief Set node description
   * @param desc Node description
   */
  void setDesc(const std::string &desc);
  
  /**
   * @brief Get node description
   * @return Node description
   */
  std::string getDesc();

  // Dynamic input/output settings
  /**
   * @brief Set whether it's dynamic input
   * @param is_dynamic_input Whether it's dynamic input
   */
  void setDynamicInput(bool is_dynamic_input);
  
  /**
   * @brief Set whether it's dynamic output
   * @param is_dynamic_output Whether it's dynamic output
   */
  void setDynamicOutput(bool is_dynamic_output);
  
  /**
   * @brief Check if it's dynamic input
   * @return Whether it's dynamic input
   */
  bool isDynamicInput();
  
  /**
   * @brief Check if it's dynamic output
   * @return Whether it's dynamic output
   */
  bool isDynamicOutput();

  // Input/output name management
  /**
   * @brief Get all input edge names
   * @return Input edge name list
   */
  std::vector<std::string> getInputNames();
  
  /**
   * @brief Get all output edge names
   * @return Output edge name list
   */
  std::vector<std::string> getOutputNames();
  
  /**
   * @brief Get input edge name at specified index
   * @param index Input edge index, default is 0
   * @return Input edge name
   */
  std::string getInputName(int index = 0);
  
  /**
   * @brief Get output edge name at specified index
   * @param index Output edge index, default is 0
   * @return Output edge name
   */
  std::string getOutputName(int index = 0);
  
  /**
   * @brief Get input edge index by name
   * @param name Input edge name
   * @return Input edge index
   */
  int getInputIndex(const std::string &name);
  
  /**
   * @brief Get output edge index by name
   * @param name Output edge name
   * @return Output edge index
   */
  int getOutputIndex(const std::string &name);
  
  /**
   * @brief Get input edge count
   * @return Input edge count
   */
  int getInputCount();
  
  /**
   * @brief Get output edge count
   * @return Output edge count
   */
  int getOutputCount();

  /**
   * @brief Set input edge name
   * @param name Input edge name
   * @param index Input edge index, default is 0
   * @return Operation status
   */
  virtual base::Status setInputName(const std::string &name, int index = 0);
  
  /**
   * @brief Set output edge name
   * @param name Output edge name
   * @param index Output edge index, default is 0
   * @return Operation status
   */
  virtual base::Status setOutputName(const std::string &name, int index = 0);
  
  /**
   * @brief Set all input edge names
   * @param names Input edge name list
   * @return Operation status
   */
  virtual base::Status setInputNames(const std::vector<std::string> &names);
  
  /**
   * @brief Set all output edge names
   * @param names Output edge name list
   * @return Operation status
   */
  virtual base::Status setOutputNames(const std::vector<std::string> &names);

  // Graph and composite node management
  /**
   * @brief Set parent graph
   * @param graph Graph pointer
   * @return Operation status
   */
  base::Status setGraph(Graph *graph);
  
  /**
   * @brief Get parent graph
   * @return Graph pointer
   */
  Graph *getGraph();
  
  /**
   * @brief Set parent composite node
   * @param composite_node Composite node pointer
   * @return Operation status
   */
  base::Status setCompositeNode(CompositeNode *composite_node);
  
  /**
   * @brief Get parent composite node
   * @return Composite node pointer
   */
  CompositeNode *getCompositeNode();

  // Device type management
  /**
   * @brief Set device type
   * @param device_type Device type
   * @return Operation status
   */
  virtual base::Status setDeviceType(base::DeviceType device_type);
  
  /**
   * @brief Get device type
   * @return Device type
   */
  virtual base::DeviceType getDeviceType();

  // Parameter management
  /**
   * @brief Set parameter
   * @param param Parameter pointer
   * @return Operation status
   */
  virtual base::Status setParam(base::Param *param);
  
  /**
   * @brief Set parameter (shared pointer)
   * @param param Parameter shared pointer
   * @return Operation status
   */
  virtual base::Status setParamSharedPtr(std::shared_ptr<base::Param> param);
  
  /**
   * @brief Get parameter
   * @return Parameter pointer
   */
  virtual base::Param *getParam();
  
  /**
   * @brief Get parameter (shared pointer)
   * @return Parameter shared pointer
   */
  virtual std::shared_ptr<base::Param> getParamSharedPtr();
  
  /**
   * @brief Set external parameter
   * @param key Parameter key
   * @param external_param External parameter shared pointer
   * @return Operation status
   */
  virtual base::Status setExternalParam(
      const std::string &key, std::shared_ptr<base::Param> external_param);
      
  /**
   * @brief Get external parameter
   * @param key Parameter key
   * @return External parameter shared pointer
   */
  virtual std::shared_ptr<base::Param> getExternalParam(const std::string &key);
  
  /**
   * @brief Set parameter (Any type)
   * @param key Parameter key
   * @param any Any type parameter
   * @return Operation status
   */
  virtual base::Status setParam(const std::string &key, base::Any &any);
  
  /**
   * @brief Get parameter (Any type)
   * @param key Parameter key
   * @param any Any type parameter reference
   * @return Operation status
   */
  virtual base::Status getParam(const std::string &key, base::Any &any);
  
  /**
   * @brief Set parameter (string type)
   * @param key Parameter key
   * @param value Parameter value
   * @return Operation status
   */
  virtual base::Status setParam(const std::string &key,
                                const std::string &value);

  // Stateless resource management
  /**
   * @brief Add stateless resource
   * @param key Resource key
   * @param value Resource value
   * @return Operation status
   */
  virtual base::Status addResourceWithoutState(const std::string &key,
                                               const base::Any &value);
                                               
  /**
   * @brief Get stateless resource
   * @param key Resource key
   * @return Resource reference
   */
  virtual base::Any &getResourceWithoutState(const std::string &key);
  
  /**
   * @brief Get stateless resource (template method)
   * @tparam T Resource type
   * @param key Resource key
   * @return Resource object
   */
  template <typename T>
  T getResourceWithoutState(const std::string &key) {
    base::Any &any = this->getResourceWithoutState(key);
    if (any.empty()) {
      // NNDEPLOY_LOGI("any is empty in getResourceWithoutState, key: %s.\n", key.c_str());
      return T();
    }
    return base::get<T>(any);
  }

  // Stateful resource management
  /**
   * @brief Create stateful resource
   * @param key Resource key
   * @return Edge pointer
   */
  virtual Edge *createResourceWithState(const std::string &key);
  
  /**
   * @brief Add stateful resource
   * @param key Resource key
   * @param edge Edge pointer
   * @return Operation status
   */
  virtual base::Status addResourceWithState(const std::string &key, Edge *edge);
  
  /**
   * @brief Get stateful resource
   * @param key Resource key
   * @return Edge pointer
   */
  virtual Edge* getResourceWithState(const std::string &key);
  
  /**
   * @brief Set stateful resource (template method)
   * @tparam T Resource type
   * @param key Resource key
   * @param value Resource value pointer
   * @param is_external Whether it's external resource, default is true
   * @return Operation status
   */
  template <typename T>
  base::Status setResourceWithState(const std::string &key, T *value, bool is_external = true) {
    Edge* edge = this->getResourceWithState(key);
    if (edge == nullptr) {
      NNDEPLOY_LOGE("edge is nullptr in setResourceWithState, key: %s.\n", key.c_str());
      return base::kStatusCodeErrorDag;
    }
    edge->set<T>(value, is_external);
    return base::kStatusCodeOk;
  }
  
  /**
   * @brief Get stateful resource (template method)
   * @tparam T Resource type
   * @param key Resource key
   * @return Resource pointer
   */
  template <typename T>
  T *getResourceWithState(const std::string &key) {
    Edge* edge = this->getResourceWithState(key);
    if (edge == nullptr) {
      NNDEPLOY_LOGE("edge is nullptr in getResourceWithState, key: %s.\n", key.c_str());
      return nullptr;
    }
    return edge->get<T>(this);
  }

  // Version management
  /**
   * @brief Set version number
   * @param version Version number
   * @return Operation status
   */
  base::Status setVersion(const std::string &version);
  
  /**
   * @brief Get version number
   * @return Version number
   */
  std::string getVersion();

  // Required parameter management
  /**
   * @brief Set required parameter list
   * @param required_params Required parameter list
   * @return Operation status
   */
  base::Status setRequiredParams(
      const std::vector<std::string> &required_params);
      
  /**
   * @brief Add required parameter
   * @param required_param Required parameter
   * @return Operation status
   */
  base::Status addRequiredParam(const std::string &required_param);
  
  /**
   * @brief Remove required parameter
   * @param required_param Required parameter
   * @return Operation status
   */
  base::Status removeRequiredParam(const std::string &required_param);
  
  /**
   * @brief Clear required parameters
   * @return Operation status
   */
  base::Status clearRequiredParams();
  
  /**
   * @brief Get required parameter list
   * @return Required parameter list
   */
  std::vector<std::string> getRequiredParams();

  // UI parameter management
  /**
   * @brief Set UI parameter list
   * @param ui_params UI parameter list
   * @return Operation status
   */
  base::Status setUiParams(const std::vector<std::string> &ui_params);
  
  /**
   * @brief Add UI parameter
   * @param ui_param UI parameter
   * @return Operation status
   */
  base::Status addUiParam(const std::string &ui_param);
  
  /**
   * @brief Remove UI parameter
   * @param ui_param UI parameter
   * @return Operation status
   */
  base::Status removeUiParam(const std::string &ui_param);
  
  /**
   * @brief Clear UI parameters
   * @return Operation status
   */
  base::Status clearUiParams();
  
  /**
   * @brief Get UI parameter list
   * @return UI parameter list
   */
  std::vector<std::string> getUiParams();

  // IO parameter management
  /**
   * @brief Set IO parameter list
   * @param io_params IO parameter list
   * @return Operation status
   */
  base::Status setIoParams(const std::vector<std::string> &io_params);
  
  /**
   * @brief Add IO parameter
   * @param io_param IO parameter
   * @return Operation status
   */
  base::Status addIoParam(const std::string &io_param);
  
  /**
   * @brief Remove IO parameter
   * @param io_param IO parameter
   * @return Operation status
   */
  base::Status removeIoParam(const std::string &io_param);
  
  /**
   * @brief Clear IO parameters
   * @return Operation status
   */
  base::Status clearIoParams();
  
  /**
   * @brief Get IO parameter list
   * @return IO parameter list
   */
  std::vector<std::string> getIoParams();

  // Dropdown parameter management
  /**
   * @brief Set dropdown parameters
   * @param dropdown_params Dropdown parameter mapping
   * @return Operation status
   */
  base::Status setDropdownParams(
      const std::map<std::string, std::vector<std::string>> &dropdown_params);
      
  /**
   * @brief Add dropdown parameter
   * @param dropdown_param Dropdown parameter name
   * @param dropdown_values Dropdown option value list
   * @return Operation status
   */
  base::Status addDropdownParam(
      const std::string &dropdown_param,
      const std::vector<std::string> &dropdown_values);
      
  /**
   * @brief Remove dropdown parameter
   * @param dropdown_param Dropdown parameter name
   * @return Operation status
   */
  base::Status removeDropdownParam(const std::string &dropdown_param);
  
  /**
   * @brief Clear dropdown parameters
   * @return Operation status
   */
  base::Status clearDropdownParams();
  
  /**
   * @brief Get dropdown parameters
   * @return Dropdown parameter mapping
   */
  std::map<std::string, std::vector<std::string>> getDropdownParams();

  // Input/output edge setting
  /**
   * @brief Set input edge
   * @param input Input edge pointer
   * @param index Input edge index, default is -1 (auto assign)
   * @return Operation status
   */
  virtual base::Status setInput(Edge *input, int index = -1);
  
  /**
   * @brief Set output edge
   * @param output Output edge pointer
   * @param index Output edge index, default is -1 (auto assign)
   * @return Operation status
   */
  virtual base::Status setOutput(Edge *output, int index = -1);
  
  /**
   * @brief Set iteration input edge
   * @param input Input edge pointer
   * @param index Input edge index, default is -1 (auto assign)
   * @return Operation status
   */
  virtual base::Status setIterInput(Edge *input, int index = -1);

  /**
   * @brief Set all input edges
   * @param inputs Input edge list
   * @return Operation status
   */
  virtual base::Status setInputs(std::vector<Edge *> inputs);
  
  /**
   * @brief Set all output edges
   * @param outputs Output edge list
   * @return Operation status
   */
  virtual base::Status setOutputs(std::vector<Edge *> outputs);

  /**
   * @brief Set input edge (shared pointer)
   * @param input Input edge shared pointer
   * @param index Input edge index, default is -1 (auto assign)
   * @return Operation status
   */
  virtual base::Status setInputSharedPtr(std::shared_ptr<Edge> input,
                                         int index = -1);
                                         
  /**
   * @brief Set output edge (shared pointer)
   * @param output Output edge shared pointer
   * @param index Output edge index, default is -1 (auto assign)
   * @return Operation status
   */
  virtual base::Status setOutputSharedPtr(std::shared_ptr<Edge> output,
                                          int index = -1);

  /**
   * @brief Set all input edges (shared pointer)
   * @param inputs Input edge shared pointer list
   * @return Operation status
   */
  virtual base::Status setInputsSharedPtr(
      std::vector<std::shared_ptr<Edge>> inputs);
      
  /**
   * @brief Set all output edges (shared pointer)
   * @param outputs Output edge shared pointer list
   * @return Operation status
   */
  virtual base::Status setOutputsSharedPtr(
      std::vector<std::shared_ptr<Edge>> outputs);

  // Input/output edge getting
  /**
   * @brief Get input edge
   * @param index Input edge index, default is 0
   * @return Input edge pointer
   */
  Edge *getInput(int index = 0);
  
  /**
   * @brief Get output edge
   * @param index Output edge index, default is 0
   * @return Output edge pointer
   */
  Edge *getOutput(int index = 0);

  /**
   * @brief Get input data (template method)
   * @tparam T Data type
   * @param index Input edge index, default is 0
   * @return Data pointer
   */
  template <typename T>
  T *getInputData(int index = 0) {
    Edge *edge = getInput(index);
    if (edge == nullptr) {
      return nullptr;
    }
    return edge->get<T>(this);
  }
  
  /**
   * @brief Set output data (template method)
   * @tparam T Data type
   * @param obj Data object pointer
   * @param index Output edge index, default is 0
   * @param is_external Whether it's external data, default is true
   * @return Operation status
   */
  template <typename T>
  base::Status setOutputData(T *obj, int index = 0, bool is_external = true) {
    Edge *edge = getOutput(index);
    if (edge == nullptr) {
      return base::kStatusCodeErrorNullParam;
    }
    return edge->set<T>(obj, is_external);
  }

  /**
   * @brief Get all input edges
   * @return Input edge list
   */
  std::vector<Edge *> getAllInput();
  
  /**
   * @brief Get all output edges
   * @return Output edge list
   */
  std::vector<Edge *> getAllOutput();

  /**
   * @brief Create internal output edge
   * @param name Edge name
   * @return Edge pointer
   */
  virtual Edge *createInternalOutputEdge(const std::string &name);

  /**
   * @brief Get whether it's constructed
   * @return Whether it's constructed
   */
  bool getConstructed();

  // Parallel type management
  /**
   * @brief Set parallel type
   * @param paralle_type Parallel type
   * @return Operation status
   */
  virtual base::Status setParallelType(const base::ParallelType &paralle_type);
  
  /**
   * @brief Get parallel type
   * @return Parallel type
   */
  virtual base::ParallelType getParallelType();

  // Flag management
  /**
   * @brief Set inner flag
   * @param flag Flag value
   */
  void setInnerFlag(bool flag);

  /**
   * @brief Set initialized flag
   * @param flag Flag value
   */
  void setInitializedFlag(bool flag);
  
  /**
   * @brief Get whether it's initialized
   * @return Whether it's initialized
   */
  bool getInitialized();

  /**
   * @brief Set time profile flag
   * @param flag Flag value
   */
  void setTimeProfileFlag(bool flag);
  
  /**
   * @brief Get time profile flag
   * @return Time profile flag
   */
  bool getTimeProfileFlag();

  /**
   * @brief Set debug flag
   * @param flag Flag value
   */
  void setDebugFlag(bool flag);
  
  /**
   * @brief Get debug flag
   * @return Debug flag
   */
  bool getDebugFlag();

  // Running state management
  /**
   * @brief Set running flag
   * @param flag Flag value
   */
  void setRunningFlag(bool flag);
  
  /**
   * @brief Check if it's running
   * @return Whether it's running
   */
  bool isRunning();
  
  /**
   * @brief Get run count
   * @return Run count
   */
  size_t getRunSize();
  
  /**
   * @brief Get completed count
   * @return Completed count
   */
  size_t getCompletedSize();
  
  /**
   * @brief Get run status
   * @return Run status shared pointer
   */
  virtual std::shared_ptr<RunStatus> getRunStatus();

  /**
   * @brief Set trace flag
   * @param flag Flag value
   */
  virtual void setTraceFlag(bool flag);
  
  /**
   * @brief Get trace flag
   * @return Trace flag
   */
  bool getTraceFlag();

  /**
   * @brief Set graph flag
   * @param flag Flag value
   */
  void setGraphFlag(bool flag);
  
  /**
   * @brief Get graph flag
   * @return Graph flag
   */
  bool getGraphFlag();

  // Node type management
  /**
   * @brief Set node type
   * @param node_type Node type
   */
  void setNodeType(NodeType node_type);
  
  /**
   * @brief Get node type
   * @return Node type
   */
  NodeType getNodeType();

  /**
   * @brief Set IO type
   * @param io_type IO type
   */
  void setIoType(IOType io_type);
  
  /**
   * @brief Get IO type
   * @return IO type
   */
  IOType getIoType();

  // Loop control
  /**
   * @brief Set loop count
   * @param loop_count Loop count
   */
  virtual void setLoopCount(int loop_count);
  
  /**
   * @brief Get loop count
   * @return Loop count
   */
  virtual int getLoopCount();

  // Stream management
  /**
   * @brief Set compute stream
   * @param stream Compute stream pointer
   */
  void setStream(device::Stream *stream);
  
  /**
   * @brief Get compute stream
   * @return Compute stream pointer
   */
  device::Stream *getStream();

  // Type information management
  /**
   * @brief Set input type information (template method)
   * @tparam T Data type
   * @param desc Description information, default is empty
   * @return Operation status
   */
  template <typename T>
  base::Status setInputTypeInfo(std::string desc = "") {
    std::shared_ptr<EdgeTypeInfo> edge_type_info =
        std::make_shared<EdgeTypeInfo>();
    edge_type_info->setType<T>();
    edge_type_info->setEdgeName(desc);
    input_type_info_.push_back(edge_type_info);
    return base::Status::Ok();
  }
  
  /**
   * @brief Set input type information
   * @param input_type_info Input type information shared pointer
   * @param desc Description information, default is empty
   * @return Operation status
   */
  base::Status setInputTypeInfo(std::shared_ptr<EdgeTypeInfo> input_type_info,
                                std::string desc = "");
                                
  /**
   * @brief Get input type information
   * @return Input type information list
   */
  std::vector<std::shared_ptr<EdgeTypeInfo>> getInputTypeInfo();

  /**
   * @brief Set output type information (template method)
   * @tparam T Data type
   * @param desc Description information, default is empty
   * @return Operation status
   */
  template <typename T>
  base::Status setOutputTypeInfo(std::string desc = "") {
    std::shared_ptr<EdgeTypeInfo> edge_type_info =
        std::make_shared<EdgeTypeInfo>();
    edge_type_info->setType<T>();
    edge_type_info->setEdgeName(desc);
    output_type_info_.push_back(edge_type_info);
    return base::Status::Ok();
  }
  
  /**
   * @brief Set output type information
   * @param output_type_info Output type information shared pointer
   * @param desc Description information, default is empty
   * @return Operation status
   */
  base::Status setOutputTypeInfo(std::shared_ptr<EdgeTypeInfo> output_type_info,
                                 std::string desc = "");
                                 
  /**
   * @brief Get output type information
   * @return Output type information list
   */
  std::vector<std::shared_ptr<EdgeTypeInfo>> getOutputTypeInfo();

  // Lifecycle management
  /**
   * @brief Configure default parameters
   * @return Configuration result status code
   */
  virtual base::Status defaultParam();

  /**
   * @brief Initialize node
   * @return Initialization status code
   */
  virtual base::Status init();
  
  /**
   * @brief Deinitialize node
   * @return Deinitialization status code
   */
  virtual base::Status deinit();

  // Memory management
  /**
   * @brief Get memory size
   * @return Memory size (bytes)
   */
  virtual int64_t getMemorySize();
  
  /**
   * @brief Set memory buffer
   * @param buffer Memory buffer pointer
   * @return Operation status
   */
  virtual base::Status setMemory(device::Buffer *buffer);

  /**
   * @brief Update input
   * @return Edge update flag
   */
  virtual base::EdgeUpdateFlag updateInput();

  // Core execution methods
  /**
   * @brief Run node (pure virtual function)
   * @return Run status code
   */
  virtual base::Status run() = 0;
  
  /**
   * @brief Synchronize execution
   * @return Whether synchronization succeeded
   */
  virtual bool synchronize();

  // Interrupt control
  /**
   * @brief Interrupt execution
   * @return Whether interruption succeeded
   */
  virtual bool interrupt();
  
  /**
   * @brief Check interrupt status
   * @return Interrupt status
   */
  virtual bool checkInterruptStatus();
  
  /**
   * @brief Clear interrupt status
   */
  virtual void clearInterrupt();

  // Forward propagation interfaces
  /**
   * @brief Node invocation interface
   * @details Node invocation interface for inter-node calls
   * @param inputs Input edges
   * @return Returned edges
   * @note
   * 1. If graph exists, return values are managed by graph
   * 2. If graph doesn't exist, return values are managed by node
   */
  virtual std::vector<Edge *> forward(std::vector<Edge *> inputs);
  
  /**
   * @brief Node invocation operator overload
   * @param inputs Input edges
   * @return Returned edges
   */
  virtual std::vector<Edge *> operator()(std::vector<Edge *> inputs);
  
  /**
   * @brief Parameter-free forward propagation
   * @return Returned edges
   */
  virtual std::vector<Edge *> forward();
  
  /**
   * @brief Parameter-free invocation operator overload
   * @return Returned edges
   */
  virtual std::vector<Edge *> operator()();
  
  /**
   * @brief Single input forward propagation
   * @param input Input edge
   * @return Returned edges
   */
  virtual std::vector<Edge *> forward(Edge *input);
  
  /**
   * @brief Single input invocation operator overload
   * @param input Input edge
   * @return Returned edges
   */
  virtual std::vector<Edge *> operator()(Edge *input);

  // Validation methods
  /**
   * @brief Check input edges
   * @param inputs Input edge list
   * @return Whether valid
   */
  bool checkInputs(std::vector<Edge *> &inputs);
  
  /**
   * @brief Check output edge names
   * @param outputs_name Output edge name list
   * @return Whether valid
   */
  bool checkOutputs(std::vector<std::string> &outputs_name);
  
  /**
   * @brief Check output edges
   * @param outputs Output edge list
   * @return Whether valid
   */
  bool checkOutputs(std::vector<Edge *> &outputs);
  
  /**
   * @brief Check if inputs changed
   * @param inputs Input edge list
   * @return Whether changed
   */
  bool isInputsChanged(std::vector<Edge *> inputs);

  /**
   * @brief Convert to static graph
   * @return Operation status
   */
  virtual base::Status toStaticGraph();

  /**
   * @brief Get real output names
   * @return Output name list
   */
  virtual std::vector<std::string> getRealOutputsName();

  // JSON serialization related methods
  /**
   * @brief Serialize to JSON
   * @param json JSON value object
   * @param allocator JSON allocator
   * @return Operation status
   */
  virtual base::Status serialize(rapidjson::Value &json,
                                 rapidjson::Document::AllocatorType &allocator);
                                 
  /**
   * @brief Serialize to JSON string
   * @return JSON string
   */
  virtual std::string serialize();
  
  /**
   * @brief Save to file
   * @param path File path
   * @return Operation status
   */
  virtual base::Status saveFile(const std::string &path);
  
  /**
   * @brief Deserialize from JSON
   * @param json JSON value object
   * @return Operation status
   */
  virtual base::Status deserialize(rapidjson::Value &json);
  
  /**
   * @brief Deserialize from JSON string
   * @param json_str JSON string
   * @return Operation status
   */
  virtual base::Status deserialize(const std::string &json_str);
  
  /**
   * @brief Load from file
   * @param path File path
   * @return Operation status
   */
  virtual base::Status loadFile(const std::string &path);

 protected:
  /**
   * @brief Node key
   * @details Node key used for node registration and creation, full type name like nndeploy::dag::Node, must be specified in constructor
   */
  std::string key_;
  std::string name_;        ///< Node name
  std::string developer_;   ///< Developer information
  std::string source_;      ///< Source information
  std::string desc_;        ///< Node description
  base::DeviceType device_type_;  ///< Device type
  
  /**
   * @brief Whether it's external stream
   * @details Indicates whether the compute stream is provided externally
   */
  bool is_external_stream_ = false;
  device::Stream *stream_ = nullptr;  ///< Compute stream pointer
  
  std::shared_ptr<base::Param> param_;  ///< Node parameters
  std::map<std::string, std::shared_ptr<base::Param>> external_param_;  ///< External parameter mapping
  
  /**
   * @brief When node input/output is dynamic, cannot set input_type_info_ and output_type_info_
   * 1. Type is uncertain
   * 2. Count is uncertain
   */
  bool is_dynamic_input_ = false;   ///< Whether it's dynamic input
  bool is_dynamic_output_ = false;  ///< Whether it's dynamic output
  
  std::vector<std::shared_ptr<EdgeTypeInfo>> input_type_info_;   ///< Input type information
  std::vector<std::shared_ptr<EdgeTypeInfo>> output_type_info_;  ///< Output type information
  std::vector<Edge *> inputs_;   ///< Input edge list
  std::vector<Edge *> outputs_;  ///< Output edge list
  std::map<std::string, Edge *> internal_outputs_;  ///< Internal output edge mapping

  Graph *graph_ = nullptr;  ///< Parent graph pointer
  CompositeNode *composite_node_ = nullptr;  ///< Parent composite node pointer

 protected:
  bool constructed_ = false;  ///< Whether constructed
  bool is_inner_ = false;     ///< Whether it's internal node in graph
  bool parallel_type_set_ = false;  ///< Whether parallel type is set
  base::ParallelType parallel_type_ = base::kParallelTypeNone;  ///< Parallel type
  bool initialized_ = false;  ///< Whether initialized
  bool is_running_ = false;   ///< Whether running
  size_t run_size_ = 0;       ///< Run count
  size_t completed_size_ = 0; ///< Completed count
  bool is_time_profile_ = false;  ///< Whether time profiling is enabled
  bool is_debug_ = false;         ///< Whether debugging is enabled
  bool is_trace_ = false;         ///< Whether tracing is enabled (when serialized to json, it must be static graph)
  bool traced_ = false;           ///< Whether traced
  bool is_graph_ = false;         ///< Whether it's graph node
  bool is_loop_ = false;          ///< Whether it's loop node
  bool is_condition_ = false;     ///< Whether it's condition node
  bool is_composite_node_ = false; ///< Whether it's composite node

  NodeType node_type_ = NodeType::kNodeTypeIntermediate;  ///< Node type
  IOType io_type_ = IOType::kIOTypeNone;  ///< IO type

  int loop_count_ = -1;  ///< Loop count
  std::atomic<bool> stop_{false};  ///< Stop flag

  std::string version_ = "1.0.0";  ///< Version number
  std::vector<std::string> required_params_;  ///< Required parameter list
  std::vector<std::string> ui_params_;        ///< UI parameter list
  std::vector<std::string> io_params_;        ///< IO parameter list
  std::map<std::string, std::vector<std::string>> dropdown_params_;  ///< Dropdown parameter mapping
};

/**
 * @brief Node creator base class
 * @details Base class for node creator used in node registration mechanism
 */
class NNDEPLOY_CC_API NodeCreator {
 public:
  /**
   * @brief Create node
   * @param node_name Node name
   * @param inputs Input edge list
   * @param outputs Output edge list
   * @return Node pointer
   */
  virtual Node *createNode(const std::string &node_name,
                           std::vector<Edge *> inputs,
                           std::vector<Edge *> outputs) = 0;
                           
  /**
   * @brief Create node (shared pointer)
   * @param node_name Node name
   * @param inputs Input edge list
   * @param outputs Output edge list
   * @return Node shared pointer
   */
  virtual std::shared_ptr<Node> createNodeSharedPtr(
      const std::string &node_name, std::vector<Edge *> inputs,
      std::vector<Edge *> outputs) = 0;
      
  virtual ~NodeCreator() = default;
};

/**
 * @brief Typed node creator
 * @tparam T Node type
 * @details Template node creator for creating specific type of nodes
 */
template <typename T>
class TypeNodeCreator : public NodeCreator {
 public:
  /**
   * @brief Create node
   * @param node_name Node name
   * @param inputs Input edge list
   * @param outputs Output edge list
   * @return Node pointer
   */
  virtual Node *createNode(const std::string &node_name,
                           std::vector<Edge *> inputs,
                           std::vector<Edge *> outputs) override {
    return new T(node_name, inputs, outputs);
  }
  
  /**
   * @brief Create node (shared pointer)
   * @param node_name Node name
   * @param inputs Input edge list
   * @param outputs Output edge list
   * @return Node shared pointer
   */
  virtual std::shared_ptr<Node> createNodeSharedPtr(
      const std::string &node_name, std::vector<Edge *> inputs,
      std::vector<Edge *> outputs) override {
    return std::make_shared<T>(node_name, inputs, outputs);
  }
};

/**
 * @brief Node factory class
 * @details Singleton node factory for managing node registration and creation
 */
class NNDEPLOY_CC_API NodeFactory {
 public:
  /**
   * @brief Get singleton instance
   * @return Node factory instance pointer
   */
  static NodeFactory *getInstance() {
    static NodeFactory instance;
    return &instance;
  }

  /**
   * @brief Register node
   * @param node_key Node key
   * @param creator Node creator shared pointer
   */
  void registerNode(const std::string &node_key,
                    std::shared_ptr<NodeCreator> creator) {
    auto it = creators_.find(node_key);
    // NNDEPLOY_LOGI("register node: %s\n", node_key.c_str());
    if (it != creators_.end()) {
      // NNDEPLOY_LOGE("Node name %s already exists!\n", node_key.c_str());
      // return;
      NNDEPLOY_LOGW("Node name %s already exists, will be overwritten!\n",
                    node_key.c_str());
    }
    creators_[node_key] = creator;
    // NNDEPLOY_LOGI("register node success: %s\n", node_key.c_str());
  }

  /**
   * @brief Get node creator
   * @param node_key Node key
   * @return Node creator shared pointer
   */
  std::shared_ptr<NodeCreator> getCreator(const std::string &node_key) {
    // for (auto &it : creators_) {
    //   NNDEPLOY_LOGI("node key: %s\n", it.first.c_str());
    // }
    auto it = creators_.find(node_key);
    if (it != creators_.end()) {
      return it->second;
    }
    return nullptr;
  }

  /**
   * @brief Get all node keys
   * @return Node key set
   */
  std::set<std::string> getNodeKeys() {
    std::set<std::string> keys;
    for (auto &it : creators_) {
      keys.insert(it.first);
    }
    return keys;
  }

 private:
  NodeFactory() = default;
  ~NodeFactory() = default;
  std::map<std::string, std::shared_ptr<NodeCreator>> creators_;  ///< Creator mapping
};

/**
 * @brief Get global node factory
 * @return Global node factory pointer
 */
extern NNDEPLOY_CC_API NodeFactory *getGlobalNodeFactory();

/**
 * @brief Node registration macro
 * @param node_key Node key
 * @param node_class Node class name
 * @details Used for automatically registering node types to factory
 */
#define REGISTER_NODE(node_key, node_class)                                \
  namespace {                                                              \
  struct NodeRegister_##node_class {                                       \
    NodeRegister_##node_class() {                                          \
      nndeploy::dag::getGlobalNodeFactory()->registerNode(                 \
          node_key,                                                        \
          std::make_shared<nndeploy::dag::TypeNodeCreator<node_class>>()); \
    }                                                                      \
  };                                                                       \
  static NodeRegister_##node_class g_node_register_##node_class;           \
  }

/**
 * @brief Get all node keys
 * @return Node key set
 */
extern NNDEPLOY_CC_API std::set<std::string> getNodeKeys();

// Node creation functions
/**
 * @brief Create node
 * @param node_key Node key
 * @param node_name Node name
 * @return Node pointer
 */
NNDEPLOY_CC_API Node *createNode(const std::string &node_key,
                                 const std::string &node_name);
                                 
/**
 * @brief Create node
 * @param node_key Node key
 * @param node_name Node name
 * @param inputs Input edge list
 * @param outputs Output edge list
 * @return Node pointer
 */
NNDEPLOY_CC_API Node *createNode(const std::string &node_key,
                                 const std::string &node_name,
                                 std::initializer_list<Edge *> inputs,
                                 std::initializer_list<Edge *> outputs);
                                 
/**
 * @brief Create node
 * @param node_key Node key
 * @param node_name Node name
 * @param inputs Input edge vector
 * @param outputs Output edge vector
 * @return Node pointer
 */
NNDEPLOY_CC_API Node *createNode(const std::string &node_key,
                                 const std::string &node_name,
                                 std::vector<Edge *> inputs,
                                 std::vector<Edge *> outputs);

/**
 * @brief Create node (shared pointer)
 * @param node_key Node key
 * @param node_name Node name
 * @return Node shared pointer
 */
NNDEPLOY_CC_API std::shared_ptr<Node> createNodeSharedPtr(
    const std::string &node_key, const std::string &node_name);
    
/**
 * @brief Create node (shared pointer)
 * @param node_key Node key
 * @param node_name Node name
 * @param inputs Input edge list
 * @param outputs Output edge list
 * @return Node shared pointer
 */
NNDEPLOY_CC_API std::shared_ptr<Node> createNodeSharedPtr(
    const std::string &node_key, const std::string &node_name,
    std::initializer_list<Edge *> inputs,
    std::initializer_list<Edge *> outputs);
    
/**
 * @brief Create node (shared pointer)
 * @param node_key Node key
 * @param node_name Node name
 * @param inputs Input edge vector
 * @param outputs Output edge vector
 * @return Node shared pointer
 */
NNDEPLOY_CC_API std::shared_ptr<Node> createNodeSharedPtr(
    const std::string &node_key, const std::string &node_name,
    std::vector<Edge *> inputs, std::vector<Edge *> outputs);

/**
 * @brief Node function type definition
 * @details Function type for functional node callbacks
 */
using NodeFunc = std::function<base::Status(std::vector<Edge *> inputs,
                                            std::vector<Edge *> outputs,
                                            base::Param *param)>;

}  // namespace dag
}  // namespace nndeploy

#endif
