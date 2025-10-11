
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

class Node;
class Graph;
class CompositeNode;

class NNDEPLOY_CC_API NodeDesc {
 public:
  NodeDesc() = default;
  NodeDesc(const std::string &node_name,
           std::initializer_list<std::string> inputs,
           std::initializer_list<std::string> outputs)
      : node_name_(node_name), inputs_(inputs), outputs_(outputs) {}
  NodeDesc(const std::string &node_name, std::vector<std::string> inputs,
           std::vector<std::string> outputs)
      : node_name_(node_name), inputs_(inputs), outputs_(outputs) {}
  NodeDesc(const std::string &node_key, const std::string &node_name,
           std::initializer_list<std::string> inputs,
           std::initializer_list<std::string> outputs)
      : node_key_(node_key),
        node_name_(node_name),
        inputs_(inputs),
        outputs_(outputs) {}
  NodeDesc(const std::string &node_key, const std::string &node_name,
           std::vector<std::string> inputs, std::vector<std::string> outputs)
      : node_key_(node_key),
        node_name_(node_name),
        inputs_(inputs),
        outputs_(outputs) {}

  virtual ~NodeDesc() = default;

  std::string getKey() const { return node_key_; }

  std::string getName() const { return node_name_; }

  std::vector<std::string> getInputs() const { return inputs_; }

  std::vector<std::string> getOutputs() const { return outputs_; }

  // to json
  base::Status serialize(rapidjson::Value &json,
                         rapidjson::Document::AllocatorType &allocator);
  std::string serialize();
  base::Status saveFile(const std::string &path);
  // from json
  base::Status deserialize(rapidjson::Value &json);
  base::Status deserialize(const std::string &json_str);
  base::Status loadFile(const std::string &path);

 private:
  // Node key
  std::string node_key_;
  // Node name
  std::string node_name_;
  // Node inputs
  std::vector<std::string> inputs_;
  // Node outputs
  std::vector<std::string> outputs_;
};

/**
 * @brief
 * @note Each node is responsible for allocating memory for it's output edges.
 */
class NNDEPLOY_CC_API Node {
 public:
  Node(const std::string &name);
  Node(const std::string &name, std::vector<Edge *> inputs,
       std::vector<Edge *> outputs);

  virtual ~Node();

  void setKey(const std::string &key);
  std::string getKey();
  void setName(const std::string &name);
  std::string getName();
  void setDeveloper(const std::string &developer);
  std::string getDeveloper();
  void setSource(const std::string &source);
  std::string getSource();
  void setDesc(const std::string &desc);
  std::string getDesc();

  void setDynamicInput(bool is_dynamic_input);
  void setDynamicOutput(bool is_dynamic_output);
  bool isDynamicInput();
  bool isDynamicOutput();

  std::vector<std::string> getInputNames();
  std::vector<std::string> getOutputNames();
  std::string getInputName(int index = 0);
  std::string getOutputName(int index = 0);
  int getInputIndex(const std::string &name);
  int getOutputIndex(const std::string &name);
  int getInputCount();
  int getOutputCount();

  virtual base::Status setInputName(const std::string &name, int index = 0);
  virtual base::Status setOutputName(const std::string &name, int index = 0);
  virtual base::Status setInputNames(const std::vector<std::string> &names);
  virtual base::Status setOutputNames(const std::vector<std::string> &names);

  base::Status setGraph(Graph *graph);
  Graph *getGraph();
  base::Status setCompositeNode(CompositeNode *composite_node);
  CompositeNode *getCompositeNode();

  virtual base::Status setDeviceType(base::DeviceType device_type);
  virtual base::DeviceType getDeviceType();

  virtual base::Status setParam(base::Param *param);
  virtual base::Status setParamSharedPtr(std::shared_ptr<base::Param> param);
  virtual base::Param *getParam();
  virtual std::shared_ptr<base::Param> getParamSharedPtr();
  virtual base::Status setExternalParam(
      const std::string &key, std::shared_ptr<base::Param> external_param);
  virtual std::shared_ptr<base::Param> getExternalParam(const std::string &key);
  virtual base::Status setParam(const std::string &key, base::Any &any);
  virtual base::Status getParam(const std::string &key, base::Any &any);
  virtual base::Status setParam(const std::string &key,
                                const std::string &value);

  virtual base::Status addResourceWithoutState(const std::string &key,
                                               const base::Any &value);
  virtual base::Any &getResourceWithoutState(const std::string &key);
  template <typename T>
  T getResourceWithoutState(const std::string &key) {
    base::Any &any = this->getResourceWithoutState(key);
    if (any.empty()) {
      NNDEPLOY_LOGI("any is empty in getResourceWithoutState, key: %s.\n", key.c_str());
      return T();
    }
    return base::get<T>(any);
  }

  virtual Edge *createResourceWithState(const std::string &key);
  virtual base::Status addResourceWithState(const std::string &key, Edge *edge);
  virtual Edge* getResourceWithState(const std::string &key);
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
  template <typename T>
  T *getResourceWithState(const std::string &key) {
    Edge* edge = this->getResourceWithState(key);
    if (edge == nullptr) {
      NNDEPLOY_LOGE("edge is nullptr in getResourceWithState, key: %s.\n", key.c_str());
      return nullptr;
    }
    return edge->get<T>(this);
  }

  base::Status setVersion(const std::string &version);
  std::string getVersion();

  base::Status setRequiredParams(
      const std::vector<std::string> &required_params);
  base::Status addRequiredParam(const std::string &required_param);
  base::Status removeRequiredParam(const std::string &required_param);
  base::Status clearRequiredParams();
  std::vector<std::string> getRequiredParams();

  base::Status setUiParams(const std::vector<std::string> &ui_params);
  base::Status addUiParam(const std::string &ui_param);
  base::Status removeUiParam(const std::string &ui_param);
  base::Status clearUiParams();
  std::vector<std::string> getUiParams();

  base::Status setIoParams(const std::vector<std::string> &io_params);
  base::Status addIoParam(const std::string &io_param);
  base::Status removeIoParam(const std::string &io_param);
  base::Status clearIoParams();
  std::vector<std::string> getIoParams();

  base::Status setDropdownParams(
      const std::map<std::string, std::vector<std::string>> &dropdown_params);
  base::Status addDropdownParam(
      const std::string &dropdown_param,
      const std::vector<std::string> &dropdown_values);
  base::Status removeDropdownParam(const std::string &dropdown_param);
  base::Status clearDropdownParams();
  std::map<std::string, std::vector<std::string>> getDropdownParams();

  virtual base::Status setInput(Edge *input, int index = -1);
  virtual base::Status setOutput(Edge *output, int index = -1);
  virtual base::Status setIterInput(Edge *input, int index = -1);

  virtual base::Status setInputs(std::vector<Edge *> inputs);
  virtual base::Status setOutputs(std::vector<Edge *> outputs);

  virtual base::Status setInputSharedPtr(std::shared_ptr<Edge> input,
                                         int index = -1);
  virtual base::Status setOutputSharedPtr(std::shared_ptr<Edge> output,
                                          int index = -1);

  virtual base::Status setInputsSharedPtr(
      std::vector<std::shared_ptr<Edge>> inputs);
  virtual base::Status setOutputsSharedPtr(
      std::vector<std::shared_ptr<Edge>> outputs);

  Edge *getInput(int index = 0);
  Edge *getOutput(int index = 0);

  template <typename T>
  T *getInputData(int index = 0) {
    Edge *edge = getInput(index);
    if (edge == nullptr) {
      return nullptr;
    }
    return edge->get<T>(this);
  }
  template <typename T>
  base::Status setOutputData(T *obj, int index = 0, bool is_external = true) {
    Edge *edge = getOutput(index);
    if (edge == nullptr) {
      return base::kStatusCodeErrorNullParam;
    }
    return edge->set<T>(obj, is_external);
  }

  std::vector<Edge *> getAllInput();
  std::vector<Edge *> getAllOutput();

  virtual Edge *createInternalOutputEdge(const std::string &name);

  bool getConstructed();

  virtual base::Status setParallelType(const base::ParallelType &paralle_type);
  virtual base::ParallelType getParallelType();

  void setInnerFlag(bool flag);

  void setInitializedFlag(bool flag);
  bool getInitialized();

  void setTimeProfileFlag(bool flag);
  bool getTimeProfileFlag();

  void setDebugFlag(bool flag);
  bool getDebugFlag();

  void setRunningFlag(bool flag);
  bool isRunning();
  size_t getRunSize();
  size_t getCompletedSize();
  virtual std::shared_ptr<RunStatus> getRunStatus();

  virtual void setTraceFlag(bool flag);
  bool getTraceFlag();

  void setGraphFlag(bool flag);
  bool getGraphFlag();

  void setNodeType(NodeType node_type);
  NodeType getNodeType();

  void setIoType(IOType io_type);
  IOType getIoType();

  virtual void setLoopCount(int loop_count);
  virtual int getLoopCount();

  void setStream(device::Stream *stream);
  device::Stream *getStream();

  template <typename T>
  base::Status setInputTypeInfo(std::string desc = "") {
    std::shared_ptr<EdgeTypeInfo> edge_type_info =
        std::make_shared<EdgeTypeInfo>();
    edge_type_info->setType<T>();
    edge_type_info->setEdgeName(desc);
    input_type_info_.push_back(edge_type_info);
    return base::Status::Ok();
  }
  base::Status setInputTypeInfo(std::shared_ptr<EdgeTypeInfo> input_type_info,
                                std::string desc = "");
  std::vector<std::shared_ptr<EdgeTypeInfo>> getInputTypeInfo();

  template <typename T>
  base::Status setOutputTypeInfo(std::string desc = "") {
    std::shared_ptr<EdgeTypeInfo> edge_type_info =
        std::make_shared<EdgeTypeInfo>();
    edge_type_info->setType<T>();
    edge_type_info->setEdgeName(desc);
    output_type_info_.push_back(edge_type_info);
    return base::Status::Ok();
  }
  base::Status setOutputTypeInfo(std::shared_ptr<EdgeTypeInfo> output_type_info,
                                 std::string desc = "");
  std::vector<std::shared_ptr<EdgeTypeInfo>> getOutputTypeInfo();

  /**
   * @brief 配置默认参数
   * @return base::Status 配置结果状态码
   */
  virtual base::Status defaultParam();

  virtual base::Status init();
  virtual base::Status deinit();

  virtual int64_t getMemorySize();
  virtual base::Status setMemory(device::Buffer *buffer);

  virtual base::EdgeUpdateFlag updateInput();

  virtual base::Status run() = 0;
  virtual bool synchronize();

  virtual bool interrupt();
  virtual bool checkInterruptStatus();
  virtual void clearInterrupt();

  /**
   * @brief 节点调用接口
   * @details 节点调用接口，用于节点之间的调用
   * @param inputs 输入的边
   * @param outputs_name 输出的边名
   * @param param 参数
   * @return 返回的边
   * @note
   * 1. 存在graph，返回值有graph管理
   * 2. 不存在graph，返回值由node管理
   */
  virtual std::vector<Edge *> forward(std::vector<Edge *> inputs);
  virtual std::vector<Edge *> operator()(std::vector<Edge *> inputs);
  virtual std::vector<Edge *> forward();
  virtual std::vector<Edge *> operator()();
  virtual std::vector<Edge *> forward(Edge *input);
  virtual std::vector<Edge *> operator()(Edge *input);

  bool checkInputs(std::vector<Edge *> &inputs);
  bool checkOutputs(std::vector<std::string> &outputs_name);
  bool checkOutputs(std::vector<Edge *> &outputs);
  bool isInputsChanged(std::vector<Edge *> inputs);

  virtual base::Status toStaticGraph();

  virtual std::vector<std::string> getRealOutputsName();

  // to json
  virtual base::Status serialize(rapidjson::Value &json,
                                 rapidjson::Document::AllocatorType &allocator);
  virtual std::string serialize();
  virtual base::Status saveFile(const std::string &path);
  // from json
  virtual base::Status deserialize(rapidjson::Value &json);
  virtual base::Status deserialize(const std::string &json_str);
  virtual base::Status loadFile(const std::string &path);

 protected:
  /**
   * @brief 节点key
   * @details
   * 节点key，用于节点注册创建，类型的全称，例如nndeploy::dag::Node，构造函数就需要指定
   */
  std::string key_;
  std::string name_;
  std::string developer_;
  std::string source_;
  std::string desc_;
  base::DeviceType device_type_;
  /**
   * @brief Whether it is an external stream
   * @details
   */
  bool is_external_stream_ = false;
  device::Stream *stream_ = nullptr;
  std::shared_ptr<base::Param> param_;
  std::map<std::string, std::shared_ptr<base::Param>> external_param_;
  /**
   * @brief 存在节点输入输出动态，无法设置input_type_info_和output_type_info_
   * 1. 类型不确定
   * 2. 个数不确定
   */
  bool is_dynamic_input_ = false;
  bool is_dynamic_output_ = false;
  std::vector<std::shared_ptr<EdgeTypeInfo>> input_type_info_;
  std::vector<std::shared_ptr<EdgeTypeInfo>> output_type_info_;
  std::vector<Edge *> inputs_;
  std::vector<Edge *> outputs_;
  std::map<std::string, Edge *> internal_outputs_;

  Graph *graph_ = nullptr;
  CompositeNode *composite_node_ = nullptr;

 protected:
  bool constructed_ = false;
  // 是否是图中内部节点
  bool is_inner_ = false;
  bool parallel_type_set_ = false;
  base::ParallelType parallel_type_ = base::kParallelTypeNone;
  bool initialized_ = false;
  bool is_running_ = false;
  size_t run_size_ = 0;
  size_t completed_size_ = 0;
  bool is_time_profile_ = false;
  bool is_debug_ = false;
  bool is_trace_ = false;  // 序列为json时，一定是静态图
  bool traced_ = false;
  bool is_graph_ = false;
  bool is_loop_ = false;
  bool is_condition_ = false;
  bool is_composite_node_ = false;

  NodeType node_type_ = NodeType::kNodeTypeIntermediate;
  IOType io_type_ = IOType::kIOTypeNone;

  int loop_count_ = -1;
  std::atomic<bool> stop_{false};

  std::string version_ = "1.0.0";
  std::vector<std::string> required_params_;
  std::vector<std::string> ui_params_;
  std::vector<std::string> io_params_;
  std::map<std::string, std::vector<std::string>> dropdown_params_;
};

/**
 * @brief 节点注册机制相关类和函数
 */
class NNDEPLOY_CC_API NodeCreator {
 public:
  virtual Node *createNode(const std::string &node_name,
                           std::vector<Edge *> inputs,
                           std::vector<Edge *> outputs) = 0;
  virtual std::shared_ptr<Node> createNodeSharedPtr(
      const std::string &node_name, std::vector<Edge *> inputs,
      std::vector<Edge *> outputs) = 0;
  virtual ~NodeCreator() = default;
};

template <typename T>
class TypeNodeCreator : public NodeCreator {
 public:
  virtual Node *createNode(const std::string &node_name,
                           std::vector<Edge *> inputs,
                           std::vector<Edge *> outputs) override {
    return new T(node_name, inputs, outputs);
  }
  virtual std::shared_ptr<Node> createNodeSharedPtr(
      const std::string &node_name, std::vector<Edge *> inputs,
      std::vector<Edge *> outputs) override {
    return std::make_shared<T>(node_name, inputs, outputs);
  }
};

class NNDEPLOY_CC_API NodeFactory {
 public:
  static NodeFactory *getInstance() {
    static NodeFactory instance;
    return &instance;
  }

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
  std::map<std::string, std::shared_ptr<NodeCreator>> creators_;
};

extern NNDEPLOY_CC_API NodeFactory *getGlobalNodeFactory();

// #define REGISTER_NODE(node_key, node_class)                              \
//   static auto register_node_creator_##node_class = []() {                \
//     nndeploy::dag::getGlobalNodeFactory()->registerNode(                 \
//         node_key,                                                        \
//         std::make_shared<nndeploy::dag::TypeNodeCreator<node_class>>()); \
//     return 0;                                                            \
//   }();

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

extern NNDEPLOY_CC_API std::set<std::string> getNodeKeys();

NNDEPLOY_CC_API Node *createNode(const std::string &node_key,
                                 const std::string &node_name);
NNDEPLOY_CC_API Node *createNode(const std::string &node_key,
                                 const std::string &node_name,
                                 std::initializer_list<Edge *> inputs,
                                 std::initializer_list<Edge *> outputs);
NNDEPLOY_CC_API Node *createNode(const std::string &node_key,
                                 const std::string &node_name,
                                 std::vector<Edge *> inputs,
                                 std::vector<Edge *> outputs);

NNDEPLOY_CC_API std::shared_ptr<Node> createNodeSharedPtr(
    const std::string &node_key, const std::string &node_name);
NNDEPLOY_CC_API std::shared_ptr<Node> createNodeSharedPtr(
    const std::string &node_key, const std::string &node_name,
    std::initializer_list<Edge *> inputs,
    std::initializer_list<Edge *> outputs);
NNDEPLOY_CC_API std::shared_ptr<Node> createNodeSharedPtr(
    const std::string &node_key, const std::string &node_name,
    std::vector<Edge *> inputs, std::vector<Edge *> outputs);

using NodeFunc = std::function<base::Status(std::vector<Edge *> inputs,
                                            std::vector<Edge *> outputs,
                                            base::Param *param)>;

}  // namespace dag
}  // namespace nndeploy

#endif
