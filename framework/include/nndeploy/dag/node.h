
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

  // NNDEPLOY_DEPRECATED("deprecated api")
  Node(const std::string &name, Edge *input, Edge *output);

  Node(const std::string &name, std::initializer_list<Edge *> inputs,
       std::initializer_list<Edge *> outputs);
  Node(const std::string &name, std::vector<Edge *> inputs,
       std::vector<Edge *> outputs);

  virtual ~Node();

  std::string getName();

  base::Status setGraph(Graph *graph) {
    graph_ = graph;
    return base::kStatusCodeOk;
  }
  Graph *getGraph() { return graph_; }

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

  void setCompiledFlag(bool flag);
  bool getCompiledFlag();

  void setStream(device::Stream *stream);
  device::Stream *getStream();

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
   * # 不在graph中
   * ## init在函数内部调用吗
   * ### 需要
   * ### 不需要
   * # 在graph中
   * ## 静态图
   * ## 半静态版动态
   * ## 动态图
   * ### 在graph中建立edge与node的关联
   * ### 第一次执行
   * ### 第二次~第n次执行
   * ### 输入edge是否更行
   */
  std::vector<std::shared_ptr<Edge>> operator()(
      std::vector<std::shared_ptr<Edge>> inputs,
      std::vector<std::string> outputs_name = std::vector<std::string>(),
      std::shared_ptr<base::Param> param = nullptr);

  // 返回内存外部管理
  std::vector<Edge *> operator()(
      std::vector<Edge *> inputs,
      std::vector<std::string> outputs_name = std::vector<std::string>(),
      std::shared_ptr<base::Param> param = nullptr);

  std::vector<std::shared_ptr<Edge>> functorWithoutGraph(
      std::vector<std::shared_ptr<Edge>> inputs,
      std::vector<std::string> outputs_name = std::vector<std::string>(),
      std::shared_ptr<base::Param> param = nullptr);

  // 返回内存外部管理
  std::vector<Edge *> functorWithoutGraph(
      std::vector<Edge *> inputs,
      std::vector<std::string> outputs_name = std::vector<std::string>(),
      std::shared_ptr<base::Param> param = nullptr);

  std::vector<std::shared_ptr<Edge>> functorWithGraph(
      std::vector<std::shared_ptr<Edge>> inputs,
      std::vector<std::string> outputs_name = std::vector<std::string>(),
      std::shared_ptr<base::Param> param = nullptr);

  // 返回内存外部管理
  std::vector<Edge *> functorWithGraph(
      std::vector<Edge *> inputs,
      std::vector<std::string> outputs_name = std::vector<std::string>(),
      std::shared_ptr<base::Param> param = nullptr);

  std::vector<std::shared_ptr<Edge>> functorDynamic(
      std::vector<std::shared_ptr<Edge>> inputs,
      std::vector<std::string> outputs_name = std::vector<std::string>(),
      std::shared_ptr<base::Param> param = nullptr);

  // 返回内存外部管理
  std::vector<Edge *> functorDynamic(
      std::vector<Edge *> inputs,
      std::vector<std::string> outputs_name = std::vector<std::string>(),
      std::shared_ptr<base::Param> param = nullptr);

  bool checkInputs(std::vector<std::shared_ptr<Edge>> &inputs);
  bool checkInputs(std::vector<Edge *> &inputs);
  bool checkOutputs(std::vector<std::string> &outputs_name);

  std::vector<std::string> getRealOutputsName(
      std::vector<std::string> outputs_name);

 protected:
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
  std::vector<EdgeTypeInfo> input_type_info_;
  std::vector<EdgeTypeInfo> output_type_info_;
  std::vector<Edge *> inputs_;
  std::vector<Edge *> outputs_;

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
  bool is_compiled_ = false;
};

/**
 * @brief 节点注册机制相关类和函数
 */
class NodeCreator {
 public:
  // virtual Node *createNode(const std::string &node_name) = 0;
  // virtual Node *createNode(const std::string &node_name,
  //                          std::initializer_list<Edge *> inputs,
  //                          std::initializer_list<Edge *> outputs) = 0;
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
  // virtual Node *createNode(const std::string &node_name) override {
  //   return new T(node_name);
  // }
  // virtual Node *createNode(const std::string &node_name,
  //                          std::initializer_list<Edge *> inputs,
  //                          std::initializer_list<Edge *> outputs) override {
  //   return new T(node_name, inputs, outputs);
  // }
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

  void registerNode(const std::string &node_key, NodeCreator *creator) {
    auto it = creators_.find(node_key);
    if (it != creators_.end()) {
      NNDEPLOY_LOGE("Node name %s already exists!\n", node_key.c_str());
      return;
    }
    creators_[node_key] = creator;
  }

  NodeCreator *getCreator(const std::string &node_key) {
    auto it = creators_.find(node_key);
    if (it != creators_.end()) {
      return it->second;
    }
    return nullptr;
  }

 private:
  NodeFactory() = default;
  ~NodeFactory() {
    for (auto &creator : creators_) {
      delete creator.second;
    }
  }
  std::map<std::string, NodeCreator *> creators_;
};

#define REGISTER_NODE(node_key, node_class)                          \
  static auto register_node_creator_##node_class = []() {            \
    nndeploy::dag::NodeFactory::getInstance()->registerNode(         \
        node_key, new nndeploy::dag::TypeNodeCreator<node_class>()); \
    return 0;                                                        \
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

using NodeFuncV1 = std::function<base::Status(
    std::initializer_list<Edge *> inputs, std::initializer_list<Edge *> outputs,
    base::Param *param)>;

using NodeFuncV2 = std::function<base::Status(std::vector<Edge *> inputs,
                                              std::vector<Edge *> outputs,
                                              base::Param *param)>;

}  // namespace dag
}  // namespace nndeploy

#endif
