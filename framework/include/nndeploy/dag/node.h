
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

class NNDEPLOY_CC_API NodeDesc {
 public:
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

  std::string getKey();
  std::string getName();

  std::vector<std::string> getInputNames();
  std::vector<std::string> getOutputNames();
  std::string getInputName(int index = 0);
  std::string getOutputName(int index = 0);
  virtual base::Status setInputName(const std::string &name, int index = 0);
  virtual base::Status setOutputName(const std::string &name, int index = 0);
  virtual base::Status setInputNames(const std::vector<std::string> &names);
  virtual base::Status setOutputNames(const std::vector<std::string> &names);

  base::Status setGraph(Graph *graph);
  Graph *getGraph();

  virtual base::Status setDeviceType(base::DeviceType device_type);
  virtual base::DeviceType getDeviceType();

  virtual base::Status setParam(base::Param *param);
  virtual base::Status setParamSharedPtr(std::shared_ptr<base::Param> param);
  virtual base::Param *getParam();
  virtual std::shared_ptr<base::Param> getParamSharedPtr();
  virtual base::Status setExternalParam(
      const std::string &key, std::shared_ptr<base::Param> external_param);
  virtual std::shared_ptr<base::Param> getExternalParam(const std::string &key);

  base::Status setInput(Edge *input, int index = -1);
  base::Status setOutput(Edge *output, int index = -1);

  base::Status setInputs(std::vector<Edge *> inputs);
  base::Status setOutputs(std::vector<Edge *> outputs);

  base::Status setInputSharedPtr(std::shared_ptr<Edge> input, int index = -1);
  base::Status setOutputSharedPtr(std::shared_ptr<Edge> output, int index = -1);

  base::Status setInputsSharedPtr(std::vector<std::shared_ptr<Edge>> inputs);
  base::Status setOutputsSharedPtr(std::vector<std::shared_ptr<Edge>> outputs);

  Edge *getInput(int index = 0);
  Edge *getOutput(int index = 0);

  std::vector<Edge *> getAllInput();
  std::vector<Edge *> getAllOutput();

  virtual Edge *createInternalOutputEdge(const std::string &name);

  bool getConstructed();

  base::Status setParallelType(const base::ParallelType &paralle_type);
  base::ParallelType getParallelType();

  void setInnerFlag(bool flag);

  void setInitializedFlag(bool flag);
  bool getInitialized();

  void setTimeProfileFlag(bool flag);
  bool getTimeProfileFlag();

  void setDebugFlag(bool flag);
  bool getDebugFlag();

  void setRunningFlag(bool flag);
  bool isRunning();

  virtual void setTraceFlag(bool flag);
  bool getTraceFlag();

  void setGraphFlag(bool flag);
  bool getGraphFlag();

  void setNodeType(NodeType node_type);
  NodeType getNodeType();

  void setStream(device::Stream *stream);
  device::Stream *getStream();

  template <typename T>
  base::Status setInputTypeInfo() {
    std::shared_ptr<EdgeTypeInfo> edge_type_info =
        std::make_shared<EdgeTypeInfo>();
    edge_type_info->setType<T>();
    input_type_info_.push_back(edge_type_info);
    return base::Status::Ok();
  }
  base::Status setInputTypeInfo(std::shared_ptr<EdgeTypeInfo> input_type_info);
  std::vector<std::shared_ptr<EdgeTypeInfo>> getInputTypeInfo();

  template <typename T>
  base::Status setOutputTypeInfo() {
    std::shared_ptr<EdgeTypeInfo> edge_type_info =
        std::make_shared<EdgeTypeInfo>();
    edge_type_info->setType<T>();
    output_type_info_.push_back(edge_type_info);
    return base::Status::Ok();
  }
  base::Status setOutputTypeInfo(
      std::shared_ptr<EdgeTypeInfo> output_type_info);
  std::vector<std::shared_ptr<EdgeTypeInfo>> getOutputTypeInfo();

  virtual base::Status init();
  virtual base::Status deinit();

  virtual int64_t getMemorySize();
  virtual base::Status setMemory(device::Buffer *buffer);

  virtual base::EdgeUpdateFlag updateInput();

  virtual base::Status run() = 0;

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

  bool checkInputs(std::vector<Edge *> &inputs);
  bool checkOutputs(std::vector<std::string> &outputs_name);
  bool checkOutputs(std::vector<Edge *> &outputs);
  bool isInputsChanged(std::vector<Edge *> inputs);

  virtual std::vector<std::string> getRealOutputsName();

  // to json
  virtual base::Status serialize(
      rapidjson::Value &json,
      rapidjson::Document::AllocatorType &allocator) const;
  virtual base::Status serialize(std::ostream &stream) const;
  virtual base::Status serialize(const std::string &path) const;
  // from json
  virtual base::Status deserialize(rapidjson::Value &json);
  virtual base::Status deserialize(std::istream &stream);
  virtual base::Status deserialize(const std::string &path);

 protected:
  /**
   * @brief 节点key
   * @details
   * 节点key，用于节点注册创建，类型的全称，例如nndeploy::dag::Node，构造函数就需要指定
   */
  std::string key_;
  std::string name_;
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
  std::vector<std::shared_ptr<EdgeTypeInfo>> input_type_info_;
  std::vector<std::shared_ptr<EdgeTypeInfo>> output_type_info_;
  std::vector<Edge *> inputs_;
  std::vector<Edge *> outputs_;
  std::map<std::string, Edge *> internal_outputs_;

  Graph *graph_ = nullptr;

 protected:
  bool constructed_ = false;
  // 是否是图中内部节点
  bool is_inner_ = false;
  base::ParallelType parallel_type_ = base::kParallelTypeNone;
  bool initialized_ = false;
  bool is_running_ = false;
  bool is_time_profile_ = false;
  bool is_debug_ = false;
  bool is_trace_ = false;
  bool traced_ = false;
  bool is_graph_ = false;
  NodeType node_type_ = NodeType::kNodeTypeIntermediate;
};

/**
 * @brief 节点注册机制相关类和函数
 */
class NodeCreator {
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

class NodeFactory {
 public:
  static NodeFactory *getInstance() {
    static NodeFactory instance;
    return &instance;
  }

  void registerNode(const std::string &node_key,
                    std::shared_ptr<NodeCreator> creator) {
    auto it = creators_.find(node_key);
    if (it != creators_.end()) {
      NNDEPLOY_LOGE("Node name %s already exists!\n", node_key.c_str());
      return;
    }
    creators_[node_key] = creator;
  }

  std::shared_ptr<NodeCreator> getCreator(const std::string &node_key) {
    auto it = creators_.find(node_key);
    if (it != creators_.end()) {
      return it->second;
    }
    return nullptr;
  }

 private:
  NodeFactory() = default;
  ~NodeFactory() = default;
  std::map<std::string, std::shared_ptr<NodeCreator>> creators_;
};

#define REGISTER_NODE(node_key, node_class)                              \
  static auto register_node_creator_##node_class = []() {                \
    nndeploy::dag::NodeFactory::getInstance()->registerNode(             \
        node_key,                                                        \
        std::make_shared<nndeploy::dag::TypeNodeCreator<node_class>>()); \
    return 0;                                                            \
  }();

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
