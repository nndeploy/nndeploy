#ifndef _NNDEPLOY_DAG_GRAPH_H_
#define _NNDEPLOY_DAG_GRAPH_H_

#include "nndeploy/base/any.h"
#include "nndeploy/base/common.h"
#include "nndeploy/base/glic_stl_include.h"
#include "nndeploy/base/log.h"
#include "nndeploy/base/macro.h"
#include "nndeploy/base/object.h"
#include "nndeploy/base/status.h"
#include "nndeploy/base/string.h"
#include "nndeploy/dag/edge.h"
#include "nndeploy/dag/executor.h"
#include "nndeploy/dag/node.h"
#include "nndeploy/dag/util.h"
#include "nndeploy/device/buffer.h"
#include "nndeploy/device/device.h"
#include "nndeploy/device/memory_pool.h"
#include "nndeploy/device/tensor.h"
#include "nndeploy/inference/inference_param.h"

namespace nndeploy {
namespace dag {

/**
 * @brief Directed Acyclic Graph Node
 * @details Graph class inherits from Node class, representing a directed acyclic graph (DAG)
 * It can contain multiple nodes and edges to build complex computation graphs
 */
class NNDEPLOY_CC_API Graph : public Node {
 public:
  /**
   * @brief Constructor
   * @param name Name of the graph
   */
  Graph(const std::string &name);
  
  /**
   * @brief Constructor
   * @param name Name of the graph
   * @param inputs List of input edges
   * @param outputs List of output edges
   */
  Graph(const std::string &name, std::vector<Edge *> inputs,
        std::vector<Edge *> outputs);
  
  /**
   * @brief Destructor
   */
  virtual ~Graph();

  /**
   * @brief Add image URL
   * @param url URL path of the image file
   * @return Operation status
   */
  base::Status addImageUrl(const std::string &url);
  
  /**
   * @brief Remove image URL
   * @param url URL path of the image file to remove
   * @return Operation status
   */
  base::Status removeImageUrl(const std::string &url);
  
  /**
   * @brief Add video URL
   * @param url URL path of the video file
   * @return Operation status
   */
  base::Status addVideoUrl(const std::string &url);
  
  /**
   * @brief Remove video URL
   * @param url URL path of the video file to remove
   * @return Operation status
   */
  base::Status removeVideoUrl(const std::string &url);
  
  /**
   * @brief Add audio URL
   * @param url URL path of the audio file
   * @return Operation status
   */
  base::Status addAudioUrl(const std::string &url);
  
  /**
   * @brief Remove audio URL
   * @param url URL path of the audio file to remove
   * @return Operation status
   */
  base::Status removeAudioUrl(const std::string &url);
  
  /**
   * @brief Add model URL
   * @param url URL path of the model file
   * @return Operation status
   */
  base::Status addModelUrl(const std::string &url);
  
  /**
   * @brief Remove model URL
   * @param url URL path of the model file to remove
   * @return Operation status
   */
  base::Status removeModelUrl(const std::string &url);
  
  /**
   * @brief Add other type URL
   * @param url URL path of other files
   * @return Operation status
   */
  base::Status addOtherUrl(const std::string &url);
  
  /**
   * @brief Remove other type URL
   * @param url URL path of other files to remove
   * @return Operation status
   */
  base::Status removeOtherUrl(const std::string &url);
  
  /**
   * @brief Get all image URL list
   * @return List of image URLs
   */
  std::vector<std::string> getImageUrl() const;
  
  /**
   * @brief Get all video URL list
   * @return List of video URLs
   */
  std::vector<std::string> getVideoUrl() const;
  
  /**
   * @brief Get all audio URL list
   * @return List of audio URLs
   */
  std::vector<std::string> getAudioUrl() const;
  
  /**
   * @brief Get all model URL list
   * @return List of model URLs
   */
  std::vector<std::string> getModelUrl() const;
  
  /**
   * @brief Get all other type URL list
   * @return List of other type URLs
   */
  std::vector<std::string> getOtherUrl() const;

  /**
   * @brief Set maximum size of edge queue
   * @param queue_max_size Maximum size of the queue
   * @return Operation status
   */
  base::Status setEdgeQueueMaxSize(int queue_max_size);
  
  /**
   * @brief Get maximum size of edge queue
   * @return Maximum size of the queue
   */
  int getEdgeQueueMaxSize();
  
  /**
   * @brief Set edge queue overflow policy
   * @param policy Overflow policy
   * @param drop_count Number of items to drop, default is 1
   * @return Operation status
   */
  base::Status setEdgeQueueOverflowPolicy(base::QueueOverflowPolicy policy,
                                          int drop_count = 1);
  
  /**
   * @brief Get edge queue overflow policy
   * @return Overflow policy
   */
  base::QueueOverflowPolicy getEdgeQueueOverflowPolicy();
  
  /**
   * @brief Get edge queue drop count
   * @return Number of items to drop
   */
  int getEdgeQueueDropCount();

  /**
   * @brief Set input edge
   * @param input Input edge pointer
   * @param index Index position, default is -1 meaning append to end
   * @return Operation status
   */
  virtual base::Status setInput(Edge *input, int index = -1);
  
  /**
   * @brief Set output edge
   * @param output Output edge pointer
   * @param index Index position, default is -1 meaning append to end
   * @return Operation status
   */
  virtual base::Status setOutput(Edge *output, int index = -1);

  /**
   * @brief Set input edge list
   * @param inputs List of input edges
   * @return Operation status
   */
  virtual base::Status setInputs(std::vector<Edge *> inputs);
  
  /**
   * @brief Set output edge list
   * @param outputs List of output edges
   * @return Operation status
   */
  virtual base::Status setOutputs(std::vector<Edge *> outputs);

  /**
   * @brief Set input edge (shared pointer version)
   * @param input Input edge shared pointer
   * @param index Index position, default is -1 meaning append to end
   * @return Operation status
   */
  virtual base::Status setInputSharedPtr(std::shared_ptr<Edge> input,
                                         int index = -1);
  
  /**
   * @brief Set output edge (shared pointer version)
   * @param output Output edge shared pointer
   * @param index Index position, default is -1 meaning append to end
   * @return Operation status
   */
  virtual base::Status setOutputSharedPtr(std::shared_ptr<Edge> output,
                                          int index = -1);

  /**
   * @brief Set input edge list (shared pointer version)
   * @param inputs List of input edge shared pointers
   * @return Operation status
   */
  virtual base::Status setInputsSharedPtr(
      std::vector<std::shared_ptr<Edge>> inputs);
  
  /**
   * @brief Set output edge list (shared pointer version)
   * @param outputs List of output edge shared pointers
   * @return Operation status
   */
  virtual base::Status setOutputsSharedPtr(
      std::vector<std::shared_ptr<Edge>> outputs);

  // Edge creation related methods
  /**
   * @brief Create edge
   * @param name Name of the edge
   * @return Created edge pointer
   */
  Edge *createEdge(const std::string &name);
  
  /**
   * @brief Create edge (shared pointer version)
   * @param name Name of the edge
   * @return Created edge shared pointer
   */
  std::shared_ptr<Edge> createEdgeSharedPtr(const std::string &name);

  // Edge addition related methods
  /**
   * @brief Add edge to graph
   * @param edge Edge pointer to add
   * @param is_external Whether it's an external edge, default is true
   * @return Edge wrapper pointer
   */
  EdgeWrapper *addEdge(Edge *edge, bool is_external = true);
  
  /**
   * @brief Add edge to graph (shared pointer version)
   * @param edge Edge shared pointer to add
   * @return Edge wrapper pointer
   */
  EdgeWrapper *addEdgeSharedPtr(std::shared_ptr<Edge> edge);

  // Edge deletion related methods
  /**
   * @brief Delete edge from graph
   * @param edge Edge pointer to delete
   * @return Operation status
   */
  base::Status deleteEdge(Edge *edge);

  // Edge retrieval related methods
  /**
   * @brief Get edge by name
   * @param name Name of the edge
   * @return Edge pointer, returns nullptr if not found
   */
  Edge *getEdge(const std::string &name);
  
  /**
   * @brief Get edge by name (shared pointer version)
   * @param name Name of the edge
   * @return Edge shared pointer, returns nullptr if not found
   */
  std::shared_ptr<Edge> getEdgeSharedPtr(const std::string &name);

  // Edge update related methods
  /**
   * @brief Update edge
   * @param edge_wrapper Edge wrapper pointer
   * @param edge New edge pointer
   * @param is_external Whether it's an external edge, default is true
   * @return Operation status
   */
  base::Status updteEdge(EdgeWrapper *edge_wrapper, Edge *edge,
                         bool is_external = true);

  // Node creation related methods
  /**
   * @brief Create node by key
   * @param key Node type key
   * @param name Node name, default is empty string
   * @return Created node pointer
   */
  Node *createNode(const std::string &key, const std::string &name = "");
  
  /**
   * @brief Create node by node description
   * @param desc Node description object
   * @return Created node pointer
   */
  Node *createNode(const NodeDesc &desc);

  /**
   * @brief Template method: Create node of specified type
   * @tparam T Node type, must inherit from Node
   * @tparam Args Constructor parameter types
   * @param name Node name, default is empty string
   * @param args Constructor parameters
   * @return Created node pointer
   */
  template <typename T, typename... Args,
            typename std::enable_if<std::is_base_of<Node, T>{}, int>::type = 0>
  Node *createNode(const std::string &name = "", Args &...args);
  
  /**
   * @brief Template method: Create node of specified type by node description
   * @tparam T Node type, must inherit from Node
   * @tparam Args Constructor parameter types
   * @param desc Node description object
   * @param args Constructor parameters
   * @return Created node pointer
   */
  template <typename T, typename... Args,
            typename std::enable_if<std::is_base_of<Node, T>{}, int>::type = 0>
  Node *createNode(const NodeDesc &desc, Args &...args);

  /**
   * @brief Set node description
   * @param node Node pointer
   * @param desc Node description object
   * @return Operation status
   */
  base::Status setNodeDesc(Node *node, const NodeDesc &desc);

  // Node addition related methods
  /**
   * @brief Add node to graph
   * @param node Node pointer to add
   * @param is_external Whether it's an external node, default is true
   * @return Operation status
   */
  base::Status addNode(Node *node, bool is_external = true);
  
  /**
   * @brief Add node to graph (shared pointer version)
   * @param node Node shared pointer to add
   * @return Operation status
   */
  base::Status addNodeSharedPtr(std::shared_ptr<Node> node);

  // Node deletion related methods
  /**
   * @brief Delete node from graph
   * @param node Node pointer to delete
   * @return Operation status
   */
  base::Status deleteNode(Node *node);

  // Node retrieval related methods
  /**
   * @brief Get node by name
   * @param name Node name
   * @return Node pointer, returns nullptr if not found
   */
  Node *getNode(const std::string &name);
  
  /**
   * @brief Get node by index
   * @param index Node index
   * @return Node pointer, returns nullptr if index is invalid
   */
  Node *getNode(int index);
  
  /**
   * @brief Get node by name (shared pointer version)
   * @param name Node name
   * @return Node shared pointer, returns nullptr if not found
   */
  std::shared_ptr<Node> getNodeSharedPtr(const std::string &name);
  
  /**
   * @brief Get node by key
   * @param key Node type key
   * @return Node pointer, returns nullptr if not found
   */
  Node *getNodeByKey(const std::string &key);
  
  /**
   * @brief Get all nodes matching the key
   * @param key Node type key
   * @return List of matching node pointers
   */
  std::vector<Node *> getNodesByKey(const std::string &key);
  
  /**
   * @brief Get total number of nodes
   * @return Number of nodes
   */
  int getNodeCount();
  
  /**
   * @brief Get all nodes
   * @return List of node pointers
   */
  std::vector<Node *> getNodes();
  
  /**
   * @brief Recursively get all nodes (including nodes in subgraphs)
   * @return List of node pointers
   */
  std::vector<Node *> getNodesRecursive();
  
  /**
   * @brief Get all node names
   * @return List of node names
   */
  std::vector<std::string> getNodesName();
  
  /**
   * @brief Recursively get all node names (including nodes in subgraphs)
   * @return List of node names
   */
  std::vector<std::string> getNodesNameRecursive();
  
  /**
   * @brief Get input node by index
   * @param index Input node index
   * @return Input node pointer
   */
  Node *getInputNode(int index);
  
  /**
   * @brief Get output node by index
   * @param index Output node index
   * @return Output node pointer
   */
  Node *getOutputNode(int index);
  
  /**
   * @brief Get inference node by index
   * @param index Inference node index
   * @return Inference node pointer
   */
  Node *getInferNode(int index);

  // Node connection related methods
  /**
   * @brief Connect two nodes
   * @param predecessor Predecessor node
   * @param successor Successor node
   * @param predecessor_port Output port of predecessor node, default is 0
   * @param successor_port Input port of successor node, default is 0
   * @return Operation status
   */
  base::Status connect(Node *predecessor, Node *successor,
                       int predecessor_port = 0, int successor_port = 0);
  
  /**
   * @brief Disconnect two nodes
   * @param predecessor Predecessor node
   * @param successor Successor node
   * @param predecessor_port Output port of predecessor node, default is 0
   * @param successor_port Input port of successor node, default is 0
   * @return Operation status
   */
  base::Status disconnect(Node *predecessor, Node *successor,
                          int predecessor_port = 0, int successor_port = 0);

  // Node run status related methods
  /**
   * @brief Get run status of all nodes
   * @return Map from node name to run status
   */
  std::map<std::string, std::shared_ptr<RunStatus>> getNodesRunStatus();
  
  /**
   * @brief Recursively get run status of all nodes (including nodes in subgraphs)
   * @return Map from node name to run status
   */
  std::map<std::string, std::shared_ptr<RunStatus>>
  getNodesRunStatusRecursive();

  // Node parameter setting related methods
  /**
   * @brief Set node parameter
   * @param node_name Node name
   * @param param Parameter pointer
   * @return Operation status
   */
  base::Status setNodeParam(const std::string &node_name, base::Param *param);
  
  /**
   * @brief Get node parameter
   * @param node_name Node name
   * @return Parameter pointer, returns nullptr if not found
   */
  base::Param *getNodeParam(const std::string &node_name);
  
  /**
   * @brief Set node parameter (shared pointer version)
   * @param node_name Node name
   * @param param Parameter shared pointer
   * @return Operation status
   */
  base::Status setNodeParamSharedPtr(const std::string &node_name,
                                     std::shared_ptr<base::Param> param);
  
  /**
   * @brief Get node parameter (shared pointer version)
   * @param node_name Node name
   * @return Parameter shared pointer, returns nullptr if not found
   */
  std::shared_ptr<base::Param> getNodeParamSharedPtr(
      const std::string &node_name);

  /**
   * @brief Set external parameter
   * @param key Parameter key
   * @param param Parameter shared pointer
   * @return Operation status
   */
  base::Status setExternalParam(const std::string &key,
                                std::shared_ptr<base::Param> param);
  
  /**
   * @brief Get external parameter
   * @param key Parameter key
   * @return Parameter shared pointer, returns nullptr if not found
   */
  std::shared_ptr<base::Param> getExternalParam(const std::string &key);

  /**
   * @brief Set node parallel type
   * @param node_name Node name
   * @param parallel_type Parallel type
   * @return Operation status
   */
  base::Status setNodeParallelType(const std::string &node_name,
                                   base::ParallelType parallel_type);

  // Graph node shared stream related methods
  /**
   * @brief Set whether graph nodes share stream
   * @param flag Flag indicating whether to share stream
   */
  void setGraphNodeShareStream(bool flag);
  
  /**
   * @brief Get whether graph nodes share stream
   * @return Flag indicating whether stream is shared
   */
  bool getGraphNodeShareStream();

  // Graph loop count related methods
  /**
   * @brief Set loop max flag
   * @param is_loop_max_flag Whether it's loop max flag
   */
  virtual void setLoopMaxFlag(bool is_loop_max_flag);
  
  /**
   * @brief Get loop max flag
   * @return Whether it's loop max flag
   */
  virtual bool getLoopMaxFlag();
  
  /**
   * @brief Set loop count
   * @param loop_count Number of loops
   */
  virtual void setLoopCount(int loop_count);
  
  /**
   * @brief Get loop count
   * @return Number of loops
   */
  virtual int getLoopCount();
  
  /**
   * @brief Get loop count map
   * @return Map from node name to loop count
   */
  virtual std::map<std::string, int> getLoopCountMap();

  // Node IO update related methods
  /**
   * @brief Update node's input and output
   * @param node Node pointer
   * @param inputs New input edge list
   * @param outputs New output edge list
   * @return Operation status
   */
  base::Status updateNodeIO(Node *node, std::vector<Edge *> inputs,
                            std::vector<Edge *> outputs);
  
  /**
   * @brief Mark input edges
   * @param inputs Input edge list
   * @return Operation status
   */
  base::Status markInputEdge(std::vector<Edge *> inputs);
  
  /**
   * @brief Mark output edges
   * @param outputs Output edge list
   * @return Operation status
   */
  base::Status markOutputEdge(std::vector<Edge *> outputs);

  /**
   * @brief Set default parameters
   * @return Operation status
   */
  virtual base::Status defaultParam();

  /**
   * @brief Initialize graph
   * @return Operation status
   */
  virtual base::Status init();
  
  /**
   * @brief Deinitialize graph
   * @return Operation status
   */
  virtual base::Status deinit();

  /**
   * @brief Run graph
   * @return Operation status
   */
  virtual base::Status run();
  
  /**
   * @brief Synchronize execution
   * @return Whether synchronization succeeded
   */
  virtual bool synchronize();
  
  /**
   * @brief Interrupt execution
   * @return Whether interruption succeeded
   */
  virtual bool interrupt();

  // Forward propagation related methods
  // These methods must be implemented by subclasses
  // Subclasses should override these methods to define their own operator() implementation
  /**
   * @brief Forward propagation (multiple inputs version)
   * @param inputs Input edge list
   * @return Output edge list
   */
  virtual std::vector<Edge *> forward(std::vector<Edge *> inputs);
  
  /**
   * @brief Operator overload (multiple inputs version)
   * @param inputs Input edge list
   * @return Output edge list
   */
  virtual std::vector<Edge *> operator()(std::vector<Edge *> inputs);
  
  /**
   * @brief Forward propagation (no input version)
   * @return Output edge list
   */
  virtual std::vector<Edge *> forward();
  
  /**
   * @brief Operator overload (no input version)
   * @return Output edge list
   */
  virtual std::vector<Edge *> operator()();
  
  /**
   * @brief Forward propagation (single input version)
   * @param input Input edge
   * @return Output edge list
   */
  virtual std::vector<Edge *> forward(Edge *input);
  
  /**
   * @brief Operator overload (single input version)
   * @param input Input edge
   * @return Output edge list
   */
  virtual std::vector<Edge *> operator()(Edge *input);

  /**
   * @brief Print graph information
   * @param oss Output stream, default is standard output
   * @return Operation status
   */
  base::Status dump(std::ostream &oss = std::cout);

  /**
   * @brief Set trace flag
   * @param flag Whether to enable tracing
   */
  virtual void setTraceFlag(bool flag);
  
  /**
   * @brief Trace execution (multiple inputs version)
   * @param inputs Input edge list
   * @return Output edge list
   */
  std::vector<Edge *> trace(std::vector<Edge *> inputs);
  
  /**
   * @brief Trace execution (no input version)
   * @return Output edge list
   */
  std::vector<Edge *> trace();
  
  /**
   * @brief Trace execution (single input version)
   * @param input Input edge
   * @return Output edge list
   */
  std::vector<Edge *> trace(Edge *input);

  /**
   * @brief Check if forward API is working properly
   * @return Whether it's working properly
   */
  bool isForwardApiOk();
  
  /**
   * @brief Convert to static graph
   * @return Operation status
   */
  virtual base::Status toStaticGraph();

  // Global resource related methods
  /**
   * @brief Add stateless global resource
   * @param key Resource key
   * @param value Resource value
   * @return Operation status
   */
  virtual base::Status addResourceWithoutState(const std::string &key,
                                               const base::Any &value);
  
  /**
   * @brief Get stateless global resource
   * @param key Resource key
   * @return Reference to resource value
   */
  virtual base::Any &getResourceWithoutState(const std::string &key);
  
  /**
   * @brief Add stateful global resource
   * @param key Resource key
   * @param edge Resource edge pointer
   * @return Operation status
   */
  virtual base::Status addResourceWithState(const std::string &key, Edge *edge);
  
  /**
   * @brief Get stateful global resource
   * @param key Resource key
   * @return Resource edge pointer
   */
  virtual Edge *getResourceWithState(const std::string &key);

  // Helper functions
  /**
   * @brief Add node's input and output
   * @param node_wrapper Node wrapper
   * @param inputs Input edge list
   * @param outputs Output edge list
   * @return Operation status
   */
  base::Status addNodeInputAndOutput(NodeWrapper *node_wrapper,
                                     std::vector<Edge *> inputs,
                                     std::vector<Edge *> outputs);

  /**
   * @brief Get edge wrapper
   * @param edge Edge pointer
   * @return Edge wrapper pointer
   */
  EdgeWrapper *getEdgeWrapper(Edge *edge);
  
  /**
   * @brief Get edge wrapper by name
   * @param name Edge name
   * @return Edge wrapper pointer
   */
  EdgeWrapper *getEdgeWrapper(const std::string &name);

  /**
   * @brief Get node wrapper
   * @param node Node pointer
   * @return Node wrapper pointer
   */
  NodeWrapper *getNodeWrapper(Node *node);
  
  /**
   * @brief Get node wrapper by name
   * @param name Node name
   * @return Node wrapper pointer
   */
  NodeWrapper *getNodeWrapper(const std::string &name);

  // JSON serialization related methods
  // using Node::serialize;
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
  
  // using Node::deserialize;
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
   * @brief Set unused node name
   * @param node_name Node name
   */
  virtual void setUnusedNodeNames(const std::string &node_name);
  
  /**
   * @brief Set unused node name set
   * @param node_names Node name set
   */
  virtual void setUnusedNodeNames(const std::set<std::string> &node_names);
  
  /**
   * @brief Remove unused node name
   * @param node_name Node name
   */
  virtual void removeUnusedNodeNames(const std::string &node_name);
  
  /**
   * @brief Remove unused node name set
   * @param node_names Node name set
   */
  virtual void removeUnusedNodeNames(const std::set<std::string> &node_names);
  
  /**
   * @brief Get unused node name set
   * @return Unused node name set
   */
  virtual std::set<std::string> getUnusedNodeNames();
  
  /**
   * @brief Remove input/output nodes
   */
  virtual void removeInOutNode();

  // Node value related methods
  // node_name:key:value
  /**
   * @brief Set node value (string format)
   * @param node_value_str Node value string in format "node_name:key:value"
   */
  virtual void setNodeValue(const std::string &node_value_str);
  
  /**
   * @brief Set node value
   * @param node_name Node name
   * @param key Key
   * @param value Value
   */
  virtual void setNodeValue(const std::string &node_name,
                            const std::string &key, const std::string &value);
  
  /**
   * @brief Set node value map
   * @param node_value_map Node value map in format {node_name: {key: value}}
   */
  virtual void setNodeValue(
      std::map<std::string, std::map<std::string, std::string>> node_value_map);
  
  /**
   * @brief Get node value map
   * @return Node value map in format {node_name: {key: value}}
   */
  virtual std::map<std::string, std::map<std::string, std::string>>
  getNodeValue();

  // Create node (deprecated API)
  /**
   * @brief Template method: Create node (single input single output)
   * @tparam T Node type, must inherit from Node
   * @tparam Args Constructor parameter types
   * @param name Node name
   * @param input Input edge
   * @param output Output edge
   * @param args Constructor parameters
   * @return Created node pointer
   */
  template <typename T, typename... Args,
            typename std::enable_if<std::is_base_of<Node, T>{}, int>::type = 0>
  Node *createNode(const std::string &name, Edge *input, Edge *output,
                   Args &...args);
  
  /**
   * @brief Template method: Create node (string input/output names)
   * @tparam T Node type, must inherit from Node
   * @tparam Args Constructor parameter types
   * @param name Node name
   * @param input_name Input edge name
   * @param output_name Output edge name
   * @param args Constructor parameters
   * @return Created node pointer
   */
  template <typename T, typename... Args,
            typename std::enable_if<std::is_base_of<Node, T>{}, int>::type = 0>
  Node *createNode(const std::string &name, const std::string &input_name,
                   const std::string &output_name, Args &...args);
  
  /**
   * @brief Template method: Create node (edge input, string output name)
   * @tparam T Node type, must inherit from Node
   * @tparam Args Constructor parameter types
   * @param name Node name
   * @param input Input edge
   * @param output_name Output edge name
   * @param args Constructor parameters
   * @return Created node pointer
   */
  template <typename T, typename... Args,
            typename std::enable_if<std::is_base_of<Node, T>{}, int>::type = 0>
  Node *createNode(const std::string &name, Edge *input,
                   const std::string &output_name, Args &...args);
  
  /**
   * @brief Template method: Create node (string input name, edge output)
   * @tparam T Node type, must inherit from Node
   * @tparam Args Constructor parameter types
   * @param name Node name
   * @param input_name Input edge name
   * @param output Output edge
   * @param args Constructor parameters
   * @return Created node pointer
   */
  template <typename T, typename... Args,
            typename std::enable_if<std::is_base_of<Node, T>{}, int>::type = 0>
  Node *createNode(const std::string &name, const std::string &input_name,
                   Edge *output, Args &...args);
  
  /**
   * @brief Template method: Create node (multiple inputs/outputs)
   * @tparam T Node type, must inherit from Node
   * @tparam Args Constructor parameter types
   * @param name Node name
   * @param inputs Input edge list
   * @param outputs Output edge list
   * @param args Constructor parameters
   * @return Created node pointer
   */
  template <typename T, typename... Args,
            typename std::enable_if<std::is_base_of<Node, T>{}, int>::type = 0>
  Node *createNode(const std::string &name, std::vector<Edge *> inputs,
                   std::vector<Edge *> outputs, Args &...args);
  
  /**
   * @brief Template method: Create node (string input/output name lists)
   * @tparam T Node type, must inherit from Node
   * @tparam Args Constructor parameter types
   * @param name Node name
   * @param input_names Input edge name list
   * @param output_names Output edge name list
   * @param args Constructor parameters
   * @return Created node pointer
   */
  template <typename T, typename... Args,
            typename std::enable_if<std::is_base_of<Node, T>{}, int>::type = 0>
  Node *createNode(const std::string &name,
                   std::vector<std::string> input_names,
                   std::vector<std::string> output_names, Args &...args);
  
  /**
   * @brief Template method: Create node (string input name list, edge output list)
   * @tparam T Node type, must inherit from Node
   * @tparam Args Constructor parameter types
   * @param name Node name
   * @param input_names Input edge name list
   * @param outputs Output edge list
   * @param args Constructor parameters
   * @return Created node pointer
   */
  template <typename T, typename... Args,
            typename std::enable_if<std::is_base_of<Node, T>{}, int>::type = 0>
  Node *createNode(const std::string &name,
                   std::vector<std::string> input_names,
                   std::vector<Edge *> outputs, Args &...args);
  
  /**
   * @brief Template method: Create node (edge input list, string output name list)
   * @tparam T Node type, must inherit from Node
   * @tparam Args Constructor parameter types
   * @param name Node name
   * @param inputs Input edge list
   * @param output_names Output edge name list
   * @param args Constructor parameters
   * @return Created node pointer
   */
  template <typename T, typename... Args,
            typename std::enable_if<std::is_base_of<Node, T>{}, int>::type = 0>
  Node *createNode(const std::string &name, std::vector<Edge *> inputs,
                   std::vector<std::string> output_names, Args &...args);
  
  /**
   * @brief Template method: Create node (initializer list inputs/outputs)
   * @tparam T Node type, must inherit from Node
   * @tparam Args Constructor parameter types
   * @param name Node name
   * @param inputs Input edge initializer list
   * @param outputs Output edge initializer list
   * @param args Constructor parameters
   * @return Created node pointer
   */
  template <typename T, typename... Args,
            typename std::enable_if<std::is_base_of<Node, T>{}, int>::type = 0>
  Node *createNode(const std::string &name,
                   std::initializer_list<Edge *> inputs,
                   std::initializer_list<Edge *> outputs, Args &...args);
  
  /**
   * @brief Template method: Create node (string initializer list input/output names)
   * @tparam T Node type, must inherit from Node
   * @tparam Args Constructor parameter types
   * @param name Node name
   * @param input_names Input edge name initializer list
   * @param output_names Output edge name initializer list
   * @param args Constructor parameters
   * @return Created node pointer
   */
  template <typename T, typename... Args,
            typename std::enable_if<std::is_base_of<Node, T>{}, int>::type = 0>
  Node *createNode(const std::string &name,
                   std::initializer_list<std::string> input_names,
                   std::initializer_list<std::string> output_names,
                   Args &...args);
  
  /**
   * @brief Template method: Create node (edge initializer list input, string initializer list output names)
   * @tparam T Node type, must inherit from Node
   * @tparam Args Constructor parameter types
   * @param name Node name
   * @param inputs Input edge initializer list
   * @param output_names Output edge name initializer list
   * @param args Constructor parameters
   * @return Created node pointer
   */
  template <typename T, typename... Args,
            typename std::enable_if<std::is_base_of<Node, T>{}, int>::type = 0>
  Node *createNode(const std::string &name,
                   std::initializer_list<Edge *> inputs,
                   std::initializer_list<std::string> output_names,
                   Args &...args);
  
  /**
   * @brief Template method: Create node (string initializer list input names, edge initializer list outputs)
   * @tparam T Node type, must inherit from Node
   * @tparam Args Constructor parameter types
   * @param name Node name
   * @param input_names Input edge name initializer list
   * @param outputs Output edge initializer list
   * @param args Constructor parameters
   * @return Created node pointer
   */
  template <typename T, typename... Args,
            typename std::enable_if<std::is_base_of<Node, T>{}, int>::type = 0>
  Node *createNode(const std::string &name,
                   std::initializer_list<std::string> input_names,
                   std::initializer_list<Edge *> outputs, Args &...args);
  
  /**
   * @brief Template method: Create inference node (single input single output)
   * @tparam T Node type, must inherit from Node
   * @tparam Args Constructor parameter types
   * @param name Node name
   * @param type Inference type
   * @param input Input edge
   * @param output Output edge
   * @return Created inference node pointer
   */
  template <typename T, typename... Args,
            typename std::enable_if<std::is_base_of<Node, T>{}, int>::type = 0>
  Node *createInfer(const std::string &name, base::InferenceType type,
                    Edge *input, Edge *output);
  
  /**
   * @brief Template method: Create inference node (string input/output names)
   * @tparam T Node type, must inherit from Node
   * @tparam Args Constructor parameter types
   * @param name Node name
   * @param type Inference type
   * @param input_name Input edge name
   * @param output_name Output edge name
   * @return Created inference node pointer
   */
  template <typename T, typename... Args,
            typename std::enable_if<std::is_base_of<Node, T>{}, int>::type = 0>
  Node *createInfer(const std::string &name, base::InferenceType type,
                    const std::string &input_name,
                    const std::string &output_name);
  
  /**
   * @brief Template method: Create inference node (edge input, string output name)
   * @tparam T Node type, must inherit from Node
   * @tparam Args Constructor parameter types
   * @param name Node name
   * @param type Inference type
   * @param input Input edge
   * @param output_name Output edge name
   * @return Created inference node pointer
   */
  template <typename T, typename... Args,
            typename std::enable_if<std::is_base_of<Node, T>{}, int>::type = 0>
  Node *createInfer(const std::string &name, base::InferenceType type,
                    Edge *input, const std::string &output_name);
  
  /**
   * @brief Template method: Create inference node (string input name, edge output)
   * @tparam T Node type, must inherit from Node
   * @tparam Args Constructor parameter types
   * @param name Node name
   * @param type Inference type
   * @param input_name Input edge name
   * @param output Output edge
   * @return Created inference node pointer
   */
  template <typename T, typename... Args,
            typename std::enable_if<std::is_base_of<Node, T>{}, int>::type = 0>
  Node *createInfer(const std::string &name, base::InferenceType type,
                    const std::string &input_name, Edge *output);
  
  /**
   * @brief Template method: Create inference node (multiple inputs/outputs)
   * @tparam T Node type, must inherit from Node
   * @tparam Args Constructor parameter types
   * @param name Node name
   * @param type Inference type
   * @param inputs Input edge list
   * @param outputs Output edge list
   * @return Created inference node pointer
   */
  template <typename T, typename... Args,
            typename std::enable_if<std::is_base_of<Node, T>{}, int>::type = 0>
  Node *createInfer(const std::string &name, base::InferenceType type,
                    std::vector<Edge *> inputs, std::vector<Edge *> outputs);
  
  /**
   * @brief Template method: Create inference node (string input/output name lists)
   * @tparam T Node type, must inherit from Node
   * @tparam Args Constructor parameter types
   * @param name Node name
   * @param type Inference type
   * @param input_names Input edge name list
   * @param output_names Output edge name list
   * @return Created inference node pointer
   */
  template <typename T, typename... Args,
            typename std::enable_if<std::is_base_of<Node, T>{}, int>::type = 0>
  Node *createInfer(const std::string &name, base::InferenceType type,
                    std::vector<std::string> input_names,
                    std::vector<std::string> output_names);
  
  /**
   * @brief Template method: Create inference node (edge input list, string output name list)
   * @tparam T Node type, must inherit from Node
   * @tparam Args Constructor parameter types
   * @param name Node name
   * @param type Inference type
   * @param inputs Input edge list
   * @param output_names Output edge name list
   * @return Created inference node pointer
   */
  template <typename T, typename... Args,
            typename std::enable_if<std::is_base_of<Node, T>{}, int>::type = 0>
  Node *createInfer(const std::string &name, base::InferenceType type,
                    std::vector<Edge *> inputs,
                    std::vector<std::string> output_names);
  
  /**
   * @brief Template method: Create inference node (string input name list, edge output list)
   * @tparam T Node type, must inherit from Node
   * @tparam Args Constructor parameter types
   * @param name Node name
   * @param type Inference type
   * @param input_names Input edge name list
   * @param outputs Output edge list
   * @return Created inference node pointer
   */
  template <typename T, typename... Args,
            typename std::enable_if<std::is_base_of<Node, T>{}, int>::type = 0>
  Node *createInfer(const std::string &name, base::InferenceType type,
                    std::vector<std::string> input_names,
                    std::vector<Edge *> outputs);
  
  /**
   * @brief Template method: Create inference node (initializer list inputs/outputs)
   * @tparam T Node type, must inherit from Node
   * @tparam Args Constructor parameter types
   * @param name Node name
   * @param type Inference type
   * @param inputs Input edge initializer list
   * @param outputs Output edge initializer list
   * @return Created inference node pointer
   */
  template <typename T, typename... Args,
            typename std::enable_if<std::is_base_of<Node, T>{}, int>::type = 0>
  Node *createInfer(const std::string &name, base::InferenceType type,
                    std::initializer_list<Edge *> inputs,
                    std::initializer_list<Edge *> outputs);
  
  /**
   * @brief Template method: Create inference node (string initializer list input/output names)
   * @tparam T Node type, must inherit from Node
   * @tparam Args Constructor parameter types
   * @param name Node name
   * @param type Inference type
   * @param input_names Input edge name initializer list
   * @param output_names Output edge name initializer list
   * @return Created inference node pointer
   */
  template <typename T, typename... Args,
            typename std::enable_if<std::is_base_of<Node, T>{}, int>::type = 0>
  Node *createInfer(const std::string &name, base::InferenceType type,
                    std::initializer_list<std::string> input_names,
                    std::initializer_list<std::string> output_names);
  
  /**
   * @brief Template method: Create inference node (edge initializer list input, string initializer list output names)
   * @tparam T Node type, must inherit from Node
   * @tparam Args Constructor parameter types
   * @param name Node name
   * @param type Inference type
   * @param inputs Input edge initializer list
   * @param output_names Output edge name initializer list
   * @return Created inference node pointer
   */
  template <typename T, typename... Args,
            typename std::enable_if<std::is_base_of<Node, T>{}, int>::type = 0>
  Node *createInfer(const std::string &name, base::InferenceType type,
                    std::initializer_list<Edge *> inputs,
                    std::initializer_list<std::string> output_names);
  
  /**
   * @brief Template method: Create inference node (string initializer list input names, edge initializer list outputs)
   * @tparam T Node type, must inherit from Node
   * @tparam Args Constructor parameter types
   * @param name Node name
   * @param type Inference type
   * @param input_names Input edge name initializer list
   * @param outputs Output edge initializer list
   * @return Created inference node pointer
   */
  template <typename T, typename... Args,
            typename std::enable_if<std::is_base_of<Node, T>{}, int>::type = 0>
  Node *createInfer(const std::string &name, base::InferenceType type,
                    std::initializer_list<std::string> input_names,
                    std::initializer_list<Edge *> outputs);
  
  /**
   * @brief Template method: Create inference node by node description
   * @tparam T Node type, must inherit from Node
   * @tparam Args Constructor parameter types
   * @param desc Node description object
   * @param type Inference type
   * @return Created inference node pointer
   */
  template <typename T, typename... Args,
            typename std::enable_if<std::is_base_of<Node, T>{}, int>::type = 0>
  Node *createInfer(const NodeDesc &desc, base::InferenceType type);
  
  /**
   * @brief Create node for Python
   * @param key Node type key
   * @param name Node name, default is empty string
   * @return Created node pointer
   */
  Node *createNode4Py(const std::string &key, const std::string &name = "");
  
  /**
   * @brief Create node for Python by node description
   * @param desc Node description object
   * @return Created node pointer
   */
  Node *createNode4Py(const NodeDesc &desc);

 protected:
  /**
   * @brief Remove unused nodes and edges
   * @return Operation status
   */
  virtual base::Status removeUnusedNodeAndEdge();
  
  /**
   * @brief Construct graph
   * @return Operation status
   */
  virtual base::Status construct();
  
  /**
   * @brief Execute graph
   * @return Operation status
   */
  virtual base::Status executor();

 protected:
  // URL storage related member variables
  std::vector<std::string> image_url_;    ///< Image URL list
  std::vector<std::string> video_url_;    ///< Video URL list
  std::vector<std::string> audio_url_;    ///< Audio URL list
  std::vector<std::string> model_url_;    ///< Model URL list
  std::vector<std::string> other_url_;    ///< Other URL list

  bool is_graph_node_share_stream_ = true;  ///< Whether graph nodes share stream
  std::vector<EdgeWrapper *> edge_repository_;        ///< Edge repository
  std::vector<NodeWrapper *> node_repository_;        ///< Node repository
  std::vector<NodeWrapper *> run_node_repository_;    ///< Run node repository
  std::vector<std::shared_ptr<Edge>> shared_edge_repository_;  ///< Shared edge repository
  std::vector<std::shared_ptr<Node>> shared_node_repository_;  ///< Shared node repository
  std::set<std::string> used_node_names_;  ///< Used node name set
  std::set<std::string> used_edge_names_;  ///< Used edge name set
  std::shared_ptr<Executor> executor_;     ///< Executor
  int queue_max_size_ = 16;                ///< Queue maximum size
  base::QueueOverflowPolicy queue_overflow_policy_ =
      base::QueueOverflowPolicy::kQueueOverflowPolicyNodeBackpressure;  ///< Queue overflow policy
  int queue_drop_count_ = 1;  ///< Queue drop count
  std::map<std::string, std::shared_ptr<base::Param>>
      external_param_repository_;  ///< External parameter repository
  bool is_loop_max_flag_ = true;    ///< Whether it's loop max flag
  bool is_forward_api_ok_ = true;   ///< Whether forward API is working properly

  bool is_remove_in_out_node_ = false;      ///< Whether to remove input/output nodes
  std::set<std::string> unused_node_names_; ///< Unused node name set
  /*
   * @brief Node values
   * @details
   * Used to store node values
   * Format: {node_name: {key: value}}
   * Note: node_name is the node name
   * Note: key is the key of the node value
   * Note: value is the value of the node value
   */
  std::map<std::string, std::map<std::string, std::string>> node_value_map_;
  /**
   * @brief Global resources (stateless)
   * @details
   * Used to store global resources
   * For example, tokenizer_encode and tokenizer_decode stages can share the same tokenizer_cpp
   */
  std::map<std::string, base::Any> resource_without_state_;
  /**
   * @brief Global resources (stateful)
   * @details
   * Used to store global resources
   * For example, history_tokens needed in the sample stage
   */
  std::map<std::string, Edge *> resource_with_state_;
};

/**
 * @brief 模板方法实现：创建指定类型的节点
 * @tparam T 节点类型，必须继承自Node
 * @tparam Args 构造函数参数类型
 * @param name 节点名称
 * @param args 构造函数参数
 * @return 创建的节点指针
 */
template <typename T, typename... Args,
          typename std::enable_if<std::is_base_of<Node, T>{}, int>::type>
Node *Graph::createNode(const std::string &name, Args &...args) {
  if (used_node_names_.find(name) != used_node_names_.end()) {
    NNDEPLOY_LOGE("node name[%s] is already used!\n", name.c_str());
    return nullptr;
  }
  std::vector<Edge *> inputs;
  std::vector<Edge *> outputs;
  Node *node = dynamic_cast<Node *>(new T(name, inputs, outputs, args...));
  NodeWrapper *node_wrapper = new NodeWrapper();
  node_wrapper->is_external_ = false;
  node_wrapper->node_ = node;
  node_wrapper->name_ = name;
  node_repository_.emplace_back(node_wrapper);
  used_node_names_.insert(name);
  node->setGraph(this);
  return node;
}

/**
 * @brief 模板方法实现：创建节点（单输入单输出）
 */
template <typename T, typename... Args,
          typename std::enable_if<std::is_base_of<Node, T>{}, int>::type>
Node *Graph::createNode(const std::string &name, Edge *input, Edge *output,
                        Args &...args) {
  if (used_node_names_.find(name) != used_node_names_.end()) {
    NNDEPLOY_LOGE("node name[%s] is already used!\n", name.c_str());
    return nullptr;
  }
  Node *node = dynamic_cast<Node *>(new T(name, {input}, {output}, args...));
  NodeWrapper *node_wrapper = new NodeWrapper();
  node_wrapper->is_external_ = false;
  node_wrapper->node_ = node;
  node_wrapper->name_ = name;
  EdgeWrapper *input_wrapper = findEdgeWrapper(edge_repository_, input);
  if (input_wrapper == nullptr) {
    input_wrapper = this->addEdge(input);
  }
  input_wrapper->consumers_.emplace_back(node_wrapper);
  EdgeWrapper *output_wrapper = findEdgeWrapper(edge_repository_, output);
  if (output_wrapper == nullptr) {
    output_wrapper = this->addEdge(output);
  }
  output_wrapper->producers_.emplace_back(node_wrapper);

  node_repository_.emplace_back(node_wrapper);
  used_node_names_.insert(name);
  node->setGraph(this);
  ;
  return node;
}

/**
 * @brief 模板方法实现：创建节点（字符串输入输出名称）
 */
template <typename T, typename... Args,
          typename std::enable_if<std::is_base_of<Node, T>{}, int>::type>
Node *Graph::createNode(const std::string &name, const std::string &input_name,
                        const std::string &output_name, Args &...args) {
  if (used_node_names_.find(name) != used_node_names_.end()) {
    NNDEPLOY_LOGE("node name[%s] is already used!\n", name.c_str());
    return nullptr;
  }
  Edge *input = getEdge(input_name);
  if (input == nullptr) {
    input = createEdge(input_name);
  }
  Edge *output = getEdge(output_name);
  if (output == nullptr) {
    output = createEdge(output_name);
  }
  Node *node = dynamic_cast<Node *>(new T(name, {input}, {output}, args...));
  NodeWrapper *node_wrapper = new NodeWrapper();
  node_wrapper->is_external_ = false;
  node_wrapper->node_ = node;
  node_wrapper->name_ = name;
  EdgeWrapper *input_wrapper = findEdgeWrapper(edge_repository_, input);
  if (input_wrapper == nullptr) {
    input_wrapper = this->addEdge(input);
  }
  input_wrapper->consumers_.emplace_back(node_wrapper);
  EdgeWrapper *output_wrapper = findEdgeWrapper(edge_repository_, output);
  if (output_wrapper == nullptr) {
    output_wrapper = this->addEdge(output);
  }
  output_wrapper->producers_.emplace_back(node_wrapper);

  node_repository_.emplace_back(node_wrapper);
  used_node_names_.insert(name);
  node->setGraph(this);
  return node;
}

/**
 * @brief 模板方法实现：创建节点（边输入，字符串输出名称）
 */
template <typename T, typename... Args,
          typename std::enable_if<std::is_base_of<Node, T>{}, int>::type>
Node *Graph::createNode(const std::string &name, Edge *input,
                        const std::string &output_name, Args &...args) {
  if (used_node_names_.find(name) != used_node_names_.end()) {
    NNDEPLOY_LOGE("node name[%s] is already used!\n", name.c_str());
    return nullptr;
  }
  Edge *output = getEdge(output_name);
  if (output == nullptr) {
    output = createEdge(output_name);
  }
  Node *node = dynamic_cast<Node *>(new T(name, {input}, {output}, args...));
  NodeWrapper *node_wrapper = new NodeWrapper();
  node_wrapper->is_external_ = false;
  node_wrapper->node_ = node;
  node_wrapper->name_ = name;
  EdgeWrapper *input_wrapper = findEdgeWrapper(edge_repository_, input);
  if (input_wrapper == nullptr) {
    input_wrapper = this->addEdge(input);
  }
  input_wrapper->consumers_.emplace_back(node_wrapper);
  EdgeWrapper *output_wrapper = findEdgeWrapper(edge_repository_, output);
  if (output_wrapper == nullptr) {
    output_wrapper = this->addEdge(output);
  }
  output_wrapper->producers_.emplace_back(node_wrapper);

  node_repository_.emplace_back(node_wrapper);
  used_node_names_.insert(name);
  node->setGraph(this);
  ;
  return node;
}

/**
 * @brief 模板方法实现：创建节点（字符串输入名称，边输出）
 */
template <typename T, typename... Args,
          typename std::enable_if<std::is_base_of<Node, T>{}, int>::type>
Node *Graph::createNode(const std::string &name, const std::string &input_name,
                        Edge *output, Args &...args) {
  if (used_node_names_.find(name) != used_node_names_.end()) {
    NNDEPLOY_LOGE("node name[%s] is already used!\n", name.c_str());
    return nullptr;
  }
  Edge *input = getEdge(input_name);
  if (input == nullptr) {
    input = createEdge(input_name);
  }
  Node *node = dynamic_cast<Node *>(new T(name, {input}, {output}, args...));
  NodeWrapper *node_wrapper = new NodeWrapper();
  node_wrapper->is_external_ = false;
  node_wrapper->node_ = node;
  node_wrapper->name_ = name;
  EdgeWrapper *input_wrapper = findEdgeWrapper(edge_repository_, input);
  if (input_wrapper == nullptr) {
    input_wrapper = this->addEdge(input);
  }
  input_wrapper->consumers_.emplace_back(node_wrapper);
  EdgeWrapper *output_wrapper = findEdgeWrapper(edge_repository_, output);
  if (output_wrapper == nullptr) {
    output_wrapper = this->addEdge(output);
  }
  output_wrapper->producers_.emplace_back(node_wrapper);

  node_repository_.emplace_back(node_wrapper);
  used_node_names_.insert(name);
  node->setGraph(this);
  ;
  return node;
}

/**
 * @brief 模板方法实现：创建节点（多输入多输出）
 */
template <typename T, typename... Args,
          typename std::enable_if<std::is_base_of<Node, T>{}, int>::type>
Node *Graph::createNode(const std::string &name, std::vector<Edge *> inputs,
                        std::vector<Edge *> outputs, Args &...args) {
  if (used_node_names_.find(name) != used_node_names_.end()) {
    NNDEPLOY_LOGE("node name[%s] is already used!\n", name.c_str());
    return nullptr;
  }
  Node *node = dynamic_cast<Node *>(new T(name, inputs, outputs, args...));
  NodeWrapper *node_wrapper = new NodeWrapper();
  node_wrapper->is_external_ = false;
  node_wrapper->node_ = node;
  node_wrapper->name_ = name;
  for (auto input : inputs) {
    EdgeWrapper *input_wrapper = findEdgeWrapper(edge_repository_, input);
    if (input_wrapper == nullptr) {
      input_wrapper = this->addEdge(input);
    }
    input_wrapper->consumers_.emplace_back(node_wrapper);
  }
  for (auto output : outputs) {
    EdgeWrapper *output_wrapper = findEdgeWrapper(edge_repository_, output);
    if (output_wrapper == nullptr) {
      output_wrapper = this->addEdge(output);
    }
    output_wrapper->producers_.emplace_back(node_wrapper);
  }

  node_repository_.emplace_back(node_wrapper);
  used_node_names_.insert(name);
  node->setGraph(this);
  ;
  return node;
}

/**
 * @brief 模板方法实现：创建节点（字符串输入输出名称列表）
 */
template <typename T, typename... Args,
          typename std::enable_if<std::is_base_of<Node, T>{}, int>::type>
Node *Graph::createNode(const std::string &name,
                        std::vector<std::string> input_names,
                        std::vector<std::string> output_names, Args &...args) {
  if (used_node_names_.find(name) != used_node_names_.end()) {
    NNDEPLOY_LOGE("node name[%s] is already used!\n", name.c_str());
    return nullptr;
  }
  std::vector<Edge *> inputs;
  for (auto input_name : input_names) {
    Edge *input = getEdge(input_name);
    if (input == nullptr) {
      input = createEdge(input_name);
    }
    inputs.emplace_back(input);
  }
  std::vector<Edge *> outputs;
  for (auto output_name : output_names) {
    Edge *output = getEdge(output_name);
    if (output == nullptr) {
      output = createEdge(output_name);
    }
    outputs.emplace_back(output);
  }
  Node *node = dynamic_cast<Node *>(new T(name, inputs, outputs, args...));
  NodeWrapper *node_wrapper = new NodeWrapper();
  node_wrapper->is_external_ = false;
  node_wrapper->node_ = node;
  node_wrapper->name_ = name;
  for (auto input : inputs) {
    EdgeWrapper *input_wrapper = findEdgeWrapper(edge_repository_, input);
    if (input_wrapper == nullptr) {
      input_wrapper = this->addEdge(input);
    }
    input_wrapper->consumers_.emplace_back(node_wrapper);
  }
  for (auto output : outputs) {
    EdgeWrapper *output_wrapper = findEdgeWrapper(edge_repository_, output);
    if (output_wrapper == nullptr) {
      output_wrapper = this->addEdge(output);
    }
    output_wrapper->producers_.emplace_back(node_wrapper);
  }

  node_repository_.emplace_back(node_wrapper);
  used_node_names_.insert(name);
  node->setGraph(this);
  return node;
}

/**
 * @brief 模板方法实现：创建节点（字符串输入名称列表，边输出列表）
 */
template <typename T, typename... Args,
          typename std::enable_if<std::is_base_of<Node, T>{}, int>::type>
Node *Graph::createNode(const std::string &name,
                        std::vector<std::string> input_names,
                        std::vector<Edge *> outputs, Args &...args) {
  if (used_node_names_.find(name) != used_node_names_.end()) {
    NNDEPLOY_LOGE("node name[%s] is already used!\n", name.c_str());
    return nullptr;
  }
  std::vector<Edge *> inputs;
  for (auto input_name : input_names) {
    Edge *input = getEdge(input_name);
    if (input == nullptr) {
      input = createEdge(input_name);
    }
    inputs.emplace_back(input);
  }
  Node *node = dynamic_cast<Node *>(new T(name, inputs, outputs, args...));
  NodeWrapper *node_wrapper = new NodeWrapper();
  node_wrapper->is_external_ = false;
  node_wrapper->node_ = node;
  node_wrapper->name_ = name;
  for (auto input : inputs) {
    EdgeWrapper *input_wrapper = findEdgeWrapper(edge_repository_, input);
    if (input_wrapper == nullptr) {
      input_wrapper = this->addEdge(input);
    }
    input_wrapper->consumers_.emplace_back(node_wrapper);
  }
  for (auto output : outputs) {
    EdgeWrapper *output_wrapper = findEdgeWrapper(edge_repository_, output);
    if (output_wrapper == nullptr) {
      output_wrapper = this->addEdge(output);
    }
    output_wrapper->producers_.emplace_back(node_wrapper);
  }

  node_repository_.emplace_back(node_wrapper);
  used_node_names_.insert(name);
  node->setGraph(this);
  return node;
}

/**
 * @brief 模板方法实现：创建节点（边输入列表，字符串输出名称列表）
 */
template <typename T, typename... Args,
          typename std::enable_if<std::is_base_of<Node, T>{}, int>::type>
Node *Graph::createNode(const std::string &name, std::vector<Edge *> inputs,
                        std::vector<std::string> output_names, Args &...args) {
  if (used_node_names_.find(name) != used_node_names_.end()) {
    NNDEPLOY_LOGE("node name[%s] is already used!\n", name.c_str());
    return nullptr;
  }
  std::vector<Edge *> outputs;
  for (auto output_name : output_names) {
    Edge *output = getEdge(output_name);
    if (output == nullptr) {
      output = createEdge(output_name);
    }
    outputs.emplace_back(output);
  }
  Node *node = dynamic_cast<Node *>(new T(name, inputs, outputs, args...));
  NodeWrapper *node_wrapper = new NodeWrapper();
  node_wrapper->is_external_ = false;
  node_wrapper->node_ = node;
  node_wrapper->name_ = name;
  for (auto input : inputs) {
    EdgeWrapper *input_wrapper = findEdgeWrapper(edge_repository_, input);
    if (input_wrapper == nullptr) {
      input_wrapper = this->addEdge(input);
    }
    input_wrapper->consumers_.emplace_back(node_wrapper);
  }
  for (auto output : outputs) {
    EdgeWrapper *output_wrapper = findEdgeWrapper(edge_repository_, output);
    if (output_wrapper == nullptr) {
      output_wrapper = this->addEdge(output);
    }
    output_wrapper->producers_.emplace_back(node_wrapper);
  }

  node_repository_.emplace_back(node_wrapper);
  used_node_names_.insert(name);
  node->setGraph(this);
  ;
  return node;
}

/**
 * @brief 模板方法实现：创建节点（初始化列表输入输出）
 */
template <typename T, typename... Args,
          typename std::enable_if<std::is_base_of<Node, T>{}, int>::type>
Node *Graph::createNode(const std::string &name,
                        std::initializer_list<Edge *> inputs,
                        std::initializer_list<Edge *> outputs, Args &...args) {
  if (used_node_names_.find(name) != used_node_names_.end()) {
    NNDEPLOY_LOGE("node name[%s] is already used!\n", name.c_str());
    return nullptr;
  }
  Node *node = dynamic_cast<Node *>(new T(name, inputs, outputs, args...));
  NodeWrapper *node_wrapper = new NodeWrapper();
  node_wrapper->is_external_ = false;
  node_wrapper->node_ = node;
  node_wrapper->name_ = name;
  for (auto input : inputs) {
    EdgeWrapper *input_wrapper = findEdgeWrapper(edge_repository_, input);
    if (input_wrapper == nullptr) {
      input_wrapper = this->addEdge(input);
    }
    input_wrapper->consumers_.emplace_back(node_wrapper);
  }
  for (auto output : outputs) {
    EdgeWrapper *output_wrapper = findEdgeWrapper(edge_repository_, output);
    if (output_wrapper == nullptr) {
      output_wrapper = this->addEdge(output);
    }
    output_wrapper->producers_.emplace_back(node_wrapper);
  }

  node_repository_.emplace_back(node_wrapper);
  used_node_names_.insert(name);
  node->setGraph(this);
  return node;
}

/**
 * @brief 模板方法实现：创建节点（字符串初始化列表输入输出名称）
 */
template <typename T, typename... Args,
          typename std::enable_if<std::is_base_of<Node, T>{}, int>::type>
Node *Graph::createNode(const std::string &name,
                        std::initializer_list<std::string> input_names,
                        std::initializer_list<std::string> output_names,
                        Args &...args) {
  if (used_node_names_.find(name) != used_node_names_.end()) {
    NNDEPLOY_LOGE("node name[%s] is already used!\n", name.c_str());
    return nullptr;
  }
  std::vector<Edge *> inputs;
  for (auto input_name : input_names) {
    Edge *input = getEdge(input_name);
    if (input == nullptr) {
      input = createEdge(input_name);
    }
    inputs.emplace_back(input);
  }
  std::vector<Edge *> outputs;
  for (auto output_name : output_names) {
    Edge *output = getEdge(output_name);
    if (output == nullptr) {
      output = createEdge(output_name);
    }
    outputs.emplace_back(output);
  }
  Node *node = dynamic_cast<Node *>(new T(name, inputs, outputs, args...));
  NodeWrapper *node_wrapper = new NodeWrapper();
  node_wrapper->is_external_ = false;
  node_wrapper->node_ = node;
  node_wrapper->name_ = name;
  for (auto input : inputs) {
    EdgeWrapper *input_wrapper = findEdgeWrapper(edge_repository_, input);
    if (input_wrapper == nullptr) {
      input_wrapper = this->addEdge(input);
    }
    input_wrapper->consumers_.emplace_back(node_wrapper);
  }
  for (auto output : outputs) {
    EdgeWrapper *output_wrapper = findEdgeWrapper(edge_repository_, output);
    if (output_wrapper == nullptr) {
      output_wrapper = this->addEdge(output);
    }
    output_wrapper->producers_.emplace_back(node_wrapper);
  }

  node_repository_.emplace_back(node_wrapper);
  used_node_names_.insert(name);
  node->setGraph(this);
  ;
  return node;
}

/**
 * @brief 模板方法实现：创建节点（边初始化列表输入，字符串初始化列表输出名称）
 */
template <typename T, typename... Args,
          typename std::enable_if<std::is_base_of<Node, T>{}, int>::type>
Node *Graph::createNode(const std::string &name,
                        std::initializer_list<Edge *> inputs,
                        std::initializer_list<std::string> output_names,
                        Args &...args) {
  if (used_node_names_.find(name) != used_node_names_.end()) {
    NNDEPLOY_LOGE("node name[%s] is already used!\n", name.c_str());
    return nullptr;
  }
  std::vector<Edge *> outputs;
  for (auto output_name : output_names) {
    Edge *output = getEdge(output_name);
    if (output == nullptr) {
      output = createEdge(output_name);
    }
    outputs.emplace_back(output);
  }
  Node *node = dynamic_cast<Node *>(new T(name, inputs, outputs, args...));
  NodeWrapper *node_wrapper = new NodeWrapper();
  node_wrapper->is_external_ = false;
  node_wrapper->node_ = node;
  node_wrapper->name_ = name;
  for (auto input : inputs) {
    EdgeWrapper *input_wrapper = findEdgeWrapper(edge_repository_, input);
    if (input_wrapper == nullptr) {
      input_wrapper = this->addEdge(input);
    }
    input_wrapper->consumers_.emplace_back(node_wrapper);
  }
  for (auto output : outputs) {
    EdgeWrapper *output_wrapper = findEdgeWrapper(edge_repository_, output);
    if (output_wrapper == nullptr) {
      output_wrapper = this->addEdge(output);
    }
    output_wrapper->producers_.emplace_back(node_wrapper);
  }

  node_repository_.emplace_back(node_wrapper);
  used_node_names_.insert(name);
  node->setGraph(this);
  ;
  return node;
}

/**
 * @brief 模板方法实现：创建节点（字符串初始化列表输入名称，边初始化列表输出）
 */
template <typename T, typename... Args,
          typename std::enable_if<std::is_base_of<Node, T>{}, int>::type>
Node *Graph::createNode(const std::string &name,
                        std::initializer_list<std::string> input_names,
                        std::initializer_list<Edge *> outputs, Args &...args) {
  if (used_node_names_.find(name) != used_node_names_.end()) {
    NNDEPLOY_LOGE("node name[%s] is already used!\n", name.c_str());
    return nullptr;
  }
  std::vector<Edge *> inputs;
  for (auto input_name : input_names) {
    Edge *input = getEdge(input_name);
    if (input == nullptr) {
      input = createEdge(input_name);
    }
    inputs.emplace_back(input);
  }
  Node *node = dynamic_cast<Node *>(new T(name, inputs, outputs, args...));
  NodeWrapper *node_wrapper = new NodeWrapper();
  node_wrapper->is_external_ = false;
  node_wrapper->node_ = node;
  node_wrapper->name_ = name;
  for (auto input : inputs) {
    EdgeWrapper *input_wrapper = findEdgeWrapper(edge_repository_, input);
    if (input_wrapper == nullptr) {
      input_wrapper = this->addEdge(input);
    }
    input_wrapper->consumers_.emplace_back(node_wrapper);
  }
  for (auto output : outputs) {
    EdgeWrapper *output_wrapper = findEdgeWrapper(edge_repository_, output);
    if (output_wrapper == nullptr) {
      output_wrapper = this->addEdge(output);
    }
    output_wrapper->producers_.emplace_back(node_wrapper);
  }

  node_repository_.emplace_back(node_wrapper);
  used_node_names_.insert(name);
  node->setGraph(this);
  ;
  return node;
}

/**
 * @brief 模板方法实现：创建推理节点（单输入单输出）
 */
template <typename T, typename... Args,
          typename std::enable_if<std::is_base_of<Node, T>{}, int>::type>
Node *Graph::createInfer(const std::string &name, base::InferenceType type,
                         Edge *input, Edge *output) {
  if (used_node_names_.find(name) != used_node_names_.end()) {
    NNDEPLOY_LOGE("node name[%s] is already used!\n", name.c_str());
    return nullptr;
  }
  Node *node = dynamic_cast<Node *>(new T(name, {input}, {output}, type));
  NodeWrapper *node_wrapper = new NodeWrapper();
  node_wrapper->is_external_ = false;
  node_wrapper->node_ = node;
  node_wrapper->name_ = name;
  EdgeWrapper *input_wrapper = findEdgeWrapper(edge_repository_, input);
  if (input_wrapper == nullptr) {
    input_wrapper = this->addEdge(input);
  }
  input_wrapper->consumers_.emplace_back(node_wrapper);
  EdgeWrapper *output_wrapper = findEdgeWrapper(edge_repository_, output);
  if (output_wrapper == nullptr) {
    output_wrapper = this->addEdge(output);
  }
  output_wrapper->producers_.emplace_back(node_wrapper);

  node_repository_.emplace_back(node_wrapper);
  used_node_names_.insert(name);
  node->setGraph(this);
  ;
  return node;
}

/**
 * @brief 模板方法实现：创建推理节点（字符串输入输出名称）
 */
template <typename T, typename... Args,
          typename std::enable_if<std::is_base_of<Node, T>{}, int>::type>
Node *Graph::createInfer(const std::string &name, base::InferenceType type,
                         const std::string &input_name,
                         const std::string &output_name) {
  if (used_node_names_.find(name) != used_node_names_.end()) {
    NNDEPLOY_LOGE("node name[%s] is already used!\n", name.c_str());
    return nullptr;
  }
  Edge *input = getEdge(input_name);
  if (input == nullptr) {
    input = createEdge(input_name);
  }
  Edge *output = getEdge(output_name);
  if (output == nullptr) {
    output = createEdge(output_name);
  }
  Node *node = dynamic_cast<Node *>(new T(name, {input}, {output}, type));
  NodeWrapper *node_wrapper = new NodeWrapper();
  node_wrapper->is_external_ = false;
  node_wrapper->node_ = node;
  node_wrapper->name_ = name;
  EdgeWrapper *input_wrapper = findEdgeWrapper(edge_repository_, input);
  if (input_wrapper == nullptr) {
    input_wrapper = this->addEdge(input);
  }
  input_wrapper->consumers_.emplace_back(node_wrapper);
  EdgeWrapper *output_wrapper = findEdgeWrapper(edge_repository_, output);
  if (output_wrapper == nullptr) {
    output_wrapper = this->addEdge(output);
  }
  output_wrapper->producers_.emplace_back(node_wrapper);

  node_repository_.emplace_back(node_wrapper);
  used_node_names_.insert(name);
  node->setGraph(this);
  ;
  return node;
}

/**
 * @brief 模板方法实现：创建推理节点（边输入，字符串输出名称）
 */
template <typename T, typename... Args,
          typename std::enable_if<std::is_base_of<Node, T>{}, int>::type>
Node *Graph::createInfer(const std::string &name, base::InferenceType type,
                         Edge *input, const std::string &output_name) {
  if (used_node_names_.find(name) != used_node_names_.end()) {
    NNDEPLOY_LOGE("node name[%s] is already used!\n", name.c_str());
    return nullptr;
  }
  Edge *output = getEdge(output_name);
  if (output == nullptr) {
    output = createEdge(output_name);
  }
  Node *node = dynamic_cast<Node *>(new T(name, {input}, {output}, type));
  NodeWrapper *node_wrapper = new NodeWrapper();
  node_wrapper->is_external_ = false;
  node_wrapper->node_ = node;
  node_wrapper->name_ = name;
  EdgeWrapper *input_wrapper = findEdgeWrapper(edge_repository_, input);
  if (input_wrapper == nullptr) {
    input_wrapper = this->addEdge(input);
  }
  input_wrapper->consumers_.emplace_back(node_wrapper);
  EdgeWrapper *output_wrapper = findEdgeWrapper(edge_repository_, output);
  if (output_wrapper == nullptr) {
    output_wrapper = this->addEdge(output);
  }
  output_wrapper->producers_.emplace_back(node_wrapper);

  node_repository_.emplace_back(node_wrapper);
  used_node_names_.insert(name);
  node->setGraph(this);
  ;
  return node;
}

/**
 * @brief 模板方法实现：创建推理节点（字符串输入名称，边输出）
 */
template <typename T, typename... Args,
          typename std::enable_if<std::is_base_of<Node, T>{}, int>::type>
Node *Graph::createInfer(const std::string &name, base::InferenceType type,
                         const std::string &input_name, Edge *output) {
  if (used_node_names_.find(name) != used_node_names_.end()) {
    NNDEPLOY_LOGE("node name[%s] is already used!\n", name.c_str());
    return nullptr;
  }
  Edge *input = getEdge(input_name);
  if (input == nullptr) {
    input = createEdge(input_name);
  }
  Node *node = dynamic_cast<Node *>(new T(name, {input}, {output}, type));
  NodeWrapper *node_wrapper = new NodeWrapper();
  node_wrapper->is_external_ = false;
  node_wrapper->node_ = node;
  node_wrapper->name_ = name;
  EdgeWrapper *input_wrapper = findEdgeWrapper(edge_repository_, input);
  if (input_wrapper == nullptr) {
    input_wrapper = this->addEdge(input);
  }
  input_wrapper->consumers_.emplace_back(node_wrapper);
  EdgeWrapper *output_wrapper = findEdgeWrapper(edge_repository_, output);
  if (output_wrapper == nullptr) {
    output_wrapper = this->addEdge(output);
  }
  output_wrapper->producers_.emplace_back(node_wrapper);

  node_repository_.emplace_back(node_wrapper);
  used_node_names_.insert(name);
  node->setGraph(this);
  ;
  return node;
}

template <typename T, typename... Args,
          typename std::enable_if<std::is_base_of<Node, T>{}, int>::type>
Node *Graph::createInfer(const std::string &name, base::InferenceType type,
                         std::vector<Edge *> inputs,
                         std::vector<Edge *> outputs) {
  if (used_node_names_.find(name) != used_node_names_.end()) {
    NNDEPLOY_LOGE("node name[%s] is already used!\n", name.c_str());
    return nullptr;
  }
  Node *node = dynamic_cast<Node *>(new T(name, inputs, outputs, type));
  NodeWrapper *node_wrapper = new NodeWrapper();
  node_wrapper->is_external_ = false;
  node_wrapper->node_ = node;
  node_wrapper->name_ = name;
  for (auto input : inputs) {
    EdgeWrapper *input_wrapper = findEdgeWrapper(edge_repository_, input);
    if (input_wrapper == nullptr) {
      input_wrapper = this->addEdge(input);
    }
    input_wrapper->consumers_.emplace_back(node_wrapper);
  }
  for (auto output : outputs) {
    EdgeWrapper *output_wrapper = findEdgeWrapper(edge_repository_, output);
    if (output_wrapper == nullptr) {
      output_wrapper = this->addEdge(output);
    }
    output_wrapper->producers_.emplace_back(node_wrapper);
  }

  node_repository_.emplace_back(node_wrapper);
  used_edge_names_.insert(name);
  node->setGraph(this);
  ;
  return node;
}

template <typename T, typename... Args,
          typename std::enable_if<std::is_base_of<Node, T>{}, int>::type>
Node *Graph::createInfer(const std::string &name, base::InferenceType type,
                         std::vector<std::string> input_names,
                         std::vector<std::string> output_names) {
  if (used_node_names_.find(name) != used_node_names_.end()) {
    NNDEPLOY_LOGE("node name[%s] is already used!\n", name.c_str());
    return nullptr;
  }
  std::vector<Edge *> inputs;
  for (auto input_name : input_names) {
    Edge *input = getEdge(input_name);
    if (input == nullptr) {
      input = createEdge(input_name);
    }
    inputs.emplace_back(input);
  }
  std::vector<Edge *> outputs;
  for (auto output_name : output_names) {
    Edge *output = getEdge(output_name);
    if (output == nullptr) {
      output = createEdge(output_name);
    }
    outputs.emplace_back(output);
  }
  Node *node = dynamic_cast<Node *>(new T(name, inputs, outputs, type));
  NodeWrapper *node_wrapper = new NodeWrapper();
  node_wrapper->is_external_ = false;
  node_wrapper->node_ = node;
  node_wrapper->name_ = name;
  for (auto input : inputs) {
    EdgeWrapper *input_wrapper = findEdgeWrapper(edge_repository_, input);
    if (input_wrapper == nullptr) {
      input_wrapper = this->addEdge(input);
    }
    input_wrapper->consumers_.emplace_back(node_wrapper);
  }
  for (auto output : outputs) {
    EdgeWrapper *output_wrapper = findEdgeWrapper(edge_repository_, output);
    if (output_wrapper == nullptr) {
      output_wrapper = this->addEdge(output);
    }
    output_wrapper->producers_.emplace_back(node_wrapper);
  }

  node_repository_.emplace_back(node_wrapper);
  used_node_names_.insert(name);
  node->setGraph(this);
  ;
  return node;
}

template <typename T, typename... Args,
          typename std::enable_if<std::is_base_of<Node, T>{}, int>::type>
Node *Graph::createInfer(const std::string &name, base::InferenceType type,
                         std::vector<Edge *> inputs,
                         std::vector<std::string> output_names) {
  if (used_node_names_.find(name) != used_node_names_.end()) {
    NNDEPLOY_LOGE("node name[%s] is already used!\n", name.c_str());
    return nullptr;
  }
  std::vector<Edge *> outputs;
  for (auto output_name : output_names) {
    Edge *output = getEdge(output_name);
    if (output == nullptr) {
      output = createEdge(output_name);
    }
    outputs.emplace_back(output);
  }
  Node *node = dynamic_cast<Node *>(new T(name, inputs, outputs, type));
  NodeWrapper *node_wrapper = new NodeWrapper();
  node_wrapper->is_external_ = false;
  node_wrapper->node_ = node;
  node_wrapper->name_ = name;
  for (auto input : inputs) {
    EdgeWrapper *input_wrapper = findEdgeWrapper(edge_repository_, input);
    if (input_wrapper == nullptr) {
      input_wrapper = this->addEdge(input);
    }
    input_wrapper->consumers_.emplace_back(node_wrapper);
  }
  for (auto output : outputs) {
    EdgeWrapper *output_wrapper = findEdgeWrapper(edge_repository_, output);
    if (output_wrapper == nullptr) {
      output_wrapper = this->addEdge(output);
    }
    output_wrapper->producers_.emplace_back(node_wrapper);
  }

  node_repository_.emplace_back(node_wrapper);
  used_node_names_.insert(name);
  node->setGraph(this);
  ;
  return node;
}

template <typename T, typename... Args,
          typename std::enable_if<std::is_base_of<Node, T>{}, int>::type>
Node *Graph::createInfer(const std::string &name, base::InferenceType type,
                         std::vector<std::string> input_names,
                         std::vector<Edge *> outputs) {
  if (used_node_names_.find(name) != used_node_names_.end()) {
    NNDEPLOY_LOGE("node name[%s] is already used!\n", name.c_str());
    return nullptr;
  }
  std::vector<Edge *> inputs;
  for (auto input_name : input_names) {
    Edge *input = getEdge(input_name);
    if (input == nullptr) {
      input = createEdge(input_name);
    }
    inputs.emplace_back(input);
  }
  Node *node = dynamic_cast<Node *>(new T(name, inputs, outputs, type));
  NodeWrapper *node_wrapper = new NodeWrapper();
  node_wrapper->is_external_ = false;
  node_wrapper->node_ = node;
  node_wrapper->name_ = name;
  for (auto input : inputs) {
    EdgeWrapper *input_wrapper = findEdgeWrapper(edge_repository_, input);
    if (input_wrapper == nullptr) {
      input_wrapper = this->addEdge(input);
    }
    input_wrapper->consumers_.emplace_back(node_wrapper);
  }
  for (auto output : outputs) {
    EdgeWrapper *output_wrapper = findEdgeWrapper(edge_repository_, output);
    if (output_wrapper == nullptr) {
      output_wrapper = this->addEdge(output);
    }
    output_wrapper->producers_.emplace_back(node_wrapper);
  }

  node_repository_.emplace_back(node_wrapper);
  used_node_names_.insert(name);
  node->setGraph(this);
  ;
  return node;
}

template <typename T, typename... Args,
          typename std::enable_if<std::is_base_of<Node, T>{}, int>::type>
Node *Graph::createInfer(const std::string &name, base::InferenceType type,
                         std::initializer_list<Edge *> inputs,
                         std::initializer_list<Edge *> outputs) {
  if (used_node_names_.find(name) != used_node_names_.end()) {
    NNDEPLOY_LOGE("node name[%s] is already used!\n", name.c_str());
    return nullptr;
  }
  Node *node = dynamic_cast<Node *>(new T(name, inputs, outputs, type));
  NodeWrapper *node_wrapper = new NodeWrapper();
  node_wrapper->is_external_ = false;
  node_wrapper->node_ = node;
  node_wrapper->name_ = name;
  for (auto input : inputs) {
    EdgeWrapper *input_wrapper = findEdgeWrapper(edge_repository_, input);
    if (input_wrapper == nullptr) {
      input_wrapper = this->addEdge(input);
    }
    input_wrapper->consumers_.emplace_back(node_wrapper);
  }
  for (auto output : outputs) {
    EdgeWrapper *output_wrapper = findEdgeWrapper(edge_repository_, output);
    if (output_wrapper == nullptr) {
      output_wrapper = this->addEdge(output);
    }
    output_wrapper->producers_.emplace_back(node_wrapper);
  }

  node_repository_.emplace_back(node_wrapper);
  used_node_names_.insert(name);
  node->setGraph(this);
  ;
  return node;
}

template <typename T, typename... Args,
          typename std::enable_if<std::is_base_of<Node, T>{}, int>::type>
Node *Graph::createInfer(const std::string &name, base::InferenceType type,
                         std::initializer_list<std::string> input_names,
                         std::initializer_list<std::string> output_names) {
  if (used_node_names_.find(name) != used_node_names_.end()) {
    NNDEPLOY_LOGE("node name[%s] is already used!\n", name.c_str());
    return nullptr;
  }
  std::vector<Edge *> inputs;
  for (auto input_name : input_names) {
    Edge *input = getEdge(input_name);
    if (input == nullptr) {
      input = createEdge(input_name);
    }
    inputs.emplace_back(input);
  }
  std::vector<Edge *> outputs;
  for (auto output_name : output_names) {
    Edge *output = getEdge(output_name);
    if (output == nullptr) {
      output = createEdge(output_name);
    }
    outputs.emplace_back(output);
  }
  Node *node = dynamic_cast<Node *>(new T(name, inputs, outputs, type));
  NodeWrapper *node_wrapper = new NodeWrapper();
  node_wrapper->is_external_ = false;
  node_wrapper->node_ = node;
  node_wrapper->name_ = name;
  for (auto input : inputs) {
    EdgeWrapper *input_wrapper = findEdgeWrapper(edge_repository_, input);
    if (input_wrapper == nullptr) {
      input_wrapper = this->addEdge(input);
    }
    input_wrapper->consumers_.emplace_back(node_wrapper);
  }
  for (auto output : outputs) {
    EdgeWrapper *output_wrapper = findEdgeWrapper(edge_repository_, output);
    if (output_wrapper == nullptr) {
      output_wrapper = this->addEdge(output);
    }
    output_wrapper->producers_.emplace_back(node_wrapper);
  }

  node_repository_.emplace_back(node_wrapper);
  used_node_names_.insert(name);
  node->setGraph(this);
  ;
  return node;
}

template <typename T, typename... Args,
          typename std::enable_if<std::is_base_of<Node, T>{}, int>::type>
Node *Graph::createInfer(const std::string &name, base::InferenceType type,
                         std::initializer_list<Edge *> inputs,
                         std::initializer_list<std::string> output_names) {
  if (used_node_names_.find(name) != used_node_names_.end()) {
    NNDEPLOY_LOGE("node name[%s] is already used!\n", name.c_str());
    return nullptr;
  }
  std::vector<Edge *> outputs;
  for (auto output_name : output_names) {
    Edge *output = getEdge(output_name);
    if (output == nullptr) {
      output = createEdge(output_name);
    }
    outputs.emplace_back(output);
  }
  Node *node = dynamic_cast<Node *>(new T(name, inputs, outputs, type));
  NodeWrapper *node_wrapper = new NodeWrapper();
  node_wrapper->is_external_ = false;
  node_wrapper->node_ = node;
  node_wrapper->name_ = name;
  for (auto input : inputs) {
    EdgeWrapper *input_wrapper = findEdgeWrapper(edge_repository_, input);
    if (input_wrapper == nullptr) {
      input_wrapper = this->addEdge(input);
    }
    input_wrapper->consumers_.emplace_back(node_wrapper);
  }
  for (auto output : outputs) {
    EdgeWrapper *output_wrapper = findEdgeWrapper(edge_repository_, output);
    if (output_wrapper == nullptr) {
      output_wrapper = this->addEdge(output);
    }
    output_wrapper->producers_.emplace_back(node_wrapper);
  }

  node_repository_.emplace_back(node_wrapper);
  used_node_names_.insert(name);
  node->setGraph(this);
  ;
  return node;
}

template <typename T, typename... Args,
          typename std::enable_if<std::is_base_of<Node, T>{}, int>::type>
Node *Graph::createInfer(const std::string &name, base::InferenceType type,
                         std::initializer_list<std::string> input_names,
                         std::initializer_list<Edge *> outputs) {
  if (used_node_names_.find(name) != used_node_names_.end()) {
    NNDEPLOY_LOGE("node name[%s] is already used!\n", name.c_str());
    return nullptr;
  }
  std::vector<Edge *> inputs;
  for (auto input_name : input_names) {
    Edge *input = getEdge(input_name);
    if (input == nullptr) {
      input = createEdge(input_name);
    }
    inputs.emplace_back(input);
  }
  std::vector<Edge *> outputs_vec;
  for (auto output : outputs) {
    outputs_vec.emplace_back(output);
  }
  Node *node = dynamic_cast<Node *>(new T(name, inputs, outputs_vec, type));
  NodeWrapper *node_wrapper = new NodeWrapper();
  node_wrapper->is_external_ = false;
  node_wrapper->node_ = node;
  node_wrapper->name_ = name;
  for (auto input : inputs) {
    EdgeWrapper *input_wrapper = findEdgeWrapper(edge_repository_, input);
    if (input_wrapper == nullptr) {
      input_wrapper = this->addEdge(input);
    }
    input_wrapper->consumers_.emplace_back(node_wrapper);
  }
  for (auto output : outputs) {
    EdgeWrapper *output_wrapper = findEdgeWrapper(edge_repository_, output);
    if (output_wrapper == nullptr) {
      output_wrapper = this->addEdge(output);
    }
    output_wrapper->producers_.emplace_back(node_wrapper);
  }

  node_repository_.emplace_back(node_wrapper);
  used_node_names_.insert(name);
  node->setGraph(this);
  ;
  return node;
}

template <typename T, typename... Args,
          typename std::enable_if<std::is_base_of<Node, T>{}, int>::type>
Node *Graph::createInfer(const NodeDesc &desc, base::InferenceType type) {
  return this->createInfer<T>(desc.getName(), type, desc.getInputs(),
                              desc.getOutputs());
}

// template <typename... Args>
// Node *Graph::createNode(const NodeDesc &desc, Args &...args) {
//   const std::string &name = desc.getName();
//   const std::string &node_key = desc.getKey();
//   std::vector<std::string> input_names = desc.getInputs();
//   std::vector<std::string> output_names = desc.getOutputs();
//   if (used_node_names_.find(name) != used_node_names_.end()) {
//     NNDEPLOY_LOGE("node name[%s] is already used!\n", name.c_str());
//     return nullptr;
//   }
//   std::vector<Edge *> inputs;
//   for (auto input_name : input_names) {
//     Edge *input = getEdge(input_name);
//     if (input == nullptr) {
//       input = createEdge(input_name);
//     }
//     inputs.emplace_back(input);
//   }
//   std::vector<Edge *> outputs;
//   for (auto output_name : output_names) {
//     Edge *output = getEdge(output_name);
//     if (output == nullptr) {
//       output = createEdge(output_name);
//     }
//     outputs.emplace_back(output);
//   }
//   Node *node = nndeploy::dag::createNode(node_key, name, inputs, outputs);
//   // Node *node =
//   //     nndeploy::dag::createNode(node_key, name, inputs, outputs, args...);
//   if (node == nullptr) {
//     NNDEPLOY_LOGE("create infer node[%s] failed!\n", desc.getName().c_str());
//     return nullptr;
//   }
//   NodeWrapper *node_wrapper = new NodeWrapper();
//   node_wrapper->is_external_ = false;
//   node_wrapper->node_ = node;
//   node_wrapper->name_ = name;
//   for (auto input : inputs) {
//     EdgeWrapper *input_wrapper = findEdgeWrapper(edge_repository_, input);
//     if (input_wrapper == nullptr) {
//       input_wrapper = this->addEdge(input);
//     }
//     input_wrapper->consumers_.emplace_back(node_wrapper);
//   }
//   for (auto output : outputs) {
//     EdgeWrapper *output_wrapper = findEdgeWrapper(edge_repository_, output);
//     if (output_wrapper == nullptr) {
//       output_wrapper = this->addEdge(output);
//     }
//     output_wrapper->producers_.emplace_back(node_wrapper);
//   }

//   node_repository_.emplace_back(node_wrapper);
//   used_node_names_.insert(name);

//   node->setGraph(this);

//   return node;
// }

// template <typename T, typename... Args,
//           typename std::enable_if<std::is_base_of<Node, T>{}, int>::type = 0>
// Node *Graph::createNode(const std::string &name = "", Args &...args) {
//   Node *node = this->createNode<T>(name, {}, {}, args...);
//   if (node == nullptr) {
//     NNDEPLOY_LOGE("create node[%s] failed!\n", name.c_str());
//     return nullptr;
//   }
//   return node;
// }
template <typename T, typename... Args,
          typename std::enable_if<std::is_base_of<Node, T>{}, int>::type>
Node *Graph::createNode(const NodeDesc &desc, Args &...args) {
  Node *node = this->createNode<T>(desc.getName(), desc.getInputs(),
                                   desc.getOutputs(), args...);
  if (node == nullptr) {
    NNDEPLOY_LOGE("create infer node[%s] failed!\n", desc.getName().c_str());
    return node;
  }
  return node;
}

// Not recommended api
using createGraphFunc = std::function<Graph *(
    const std::string &name, base::InferenceType inference_type,
    base::DeviceType device_type, Edge *input, Edge *output,
    base::ModelType model_type, bool is_path,
    std::vector<std::string> model_value)>;

extern NNDEPLOY_CC_API std::map<std::string, createGraphFunc> &
getGlobalGraphCreatorMap();

class NNDEPLOY_CC_API TypeGraphRegister{public : explicit TypeGraphRegister(
    const std::string &name,
    createGraphFunc func){getGlobalGraphCreatorMap()[name] = func;
}  // namespace dag
};  // namespace nndeploy

extern NNDEPLOY_CC_API Graph *createGraph(const std::string &name,
                                          base::InferenceType inference_type,
                                          base::DeviceType device_type,
                                          Edge *input, Edge *output,
                                          base::ModelType model_type,
                                          bool is_path,
                                          std::vector<std::string> model_value);

// to json
extern NNDEPLOY_CC_API base::Status serialize(
    Graph *graph, rapidjson::Value &json,
    rapidjson::Document::AllocatorType &allocator);
extern NNDEPLOY_CC_API std::string serialize(Graph *graph);
extern NNDEPLOY_CC_API base::Status saveFile(Graph *graph,
                                             const std::string &path);
// from json
extern NNDEPLOY_CC_API Graph *deserialize(rapidjson::Value &json);
extern NNDEPLOY_CC_API Graph *deserialize(const std::string &json_str);
extern NNDEPLOY_CC_API Graph *loadFile(const std::string &path);

}  // namespace dag
}  // namespace nndeploy

#endif  // _NNDEPLOY_DAG_EXECUTOR_H_
