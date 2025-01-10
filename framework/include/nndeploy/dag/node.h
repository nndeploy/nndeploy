
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
#include "nndeploy/dag/edge.h"
#include "nndeploy/device/buffer.h"
#include "nndeploy/device/device.h"
#include "nndeploy/device/memory_pool.h"
#include "nndeploy/device/tensor.h"

namespace nndeploy {
namespace dag {

class Node;
class Graph;

template <typename T_PARAM,
          typename std::enable_if<std::is_base_of<base::Param, T_PARAM>{},
                                  int>::type = 0>
class NNDEPLOY_CC_API NodeDesc {
 public:
  // Node name
  std::string name_;
  // Node inputs
  std::vector<std::string> inputs_;
  // Node outputs
  std::vector<std::string> outputs_;
  // Node parameters
  std::shared_ptr<T_PARAM> node_param_;
};

/**
 * @brief
 * @note Each node is responsible for allocating memory for it's output edges.
 */
class NNDEPLOY_CC_API Node {
 public:
  Node(const std::string &name);
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

  base::Status setDeviceType(base::DeviceType device_type);
  base::DeviceType getDeviceType();

  virtual base::Status setParam(base::Param *param);
  virtual base::Param *getParam();
  virtual base::Status setExternalParam(base::Param *external_param);

  base::Status setInput(Edge *input);
  base::Status setOutput(Edge *output);

  base::Status setInputs(std::vector<Edge *> inputs);
  base::Status setOutputs(std::vector<Edge *> outputs);

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

  virtual base::Status init();
  virtual base::Status deinit();

  virtual int64_t getMemorySize();
  virtual base::Status setMemory(device::Buffer *buffer);

  virtual base::EdgeUpdateFlag updataInput();

  virtual base::Status run() = 0;

  virtual std::vector<Edge *> operator()(
      std::vector<Edge *> inputs,
      std::vector<std::string> outputs_name = std::vector<std::string>());

  virtual std::vector<Edge *> operator()(
      std::initializer_list<Edge *> inputs,
      std::initializer_list<std::string> outputs_name = {});

 protected:
  std::string name_;
  base::DeviceType device_type_;
  std::shared_ptr<base::Param> param_;
  std::vector<base::Param *> external_param_;
  std::vector<Edge *> inputs_;
  std::vector<Edge *> outputs_;

  Graph *graph_ = nullptr;

  bool constructed_ = false;
  // 是否是图中内部节点
  bool is_inner_ = false;
  base::ParallelType parallel_type_ = base::kParallelTypeNone;
  bool initialized_ = false;
  bool is_running_ = false;
  bool is_time_profile_ = false;
  bool is_debug_ = false;
};

/**
 * @brief 节点注册机制相关类和函数
 */
class NodeCreator {
 public:
  virtual Node *createNode() = 0;
  virtual ~NodeCreator() = default;
};

template <typename T>
class NodeCreatorRegister : public NodeCreator {
 public:
  Node *createNode() override { return new T(); }
};

class NodeFactory {
 public:
  static NodeFactory *getInstance() {
    static NodeFactory instance;
    return &instance;
  }

  void registerNode(const std::string &name, NodeCreator *creator) {
    auto it = creators_.find(name);
    if (it != creators_.end()) {
      NNDEPLOY_LOGE("Node name %s already exists!\n", name.c_str());
      return;
    }
    creators_[name] = creator;
  }

  Node *createNode(const std::string &name) {
    auto it = creators_.find(name);
    if (it != creators_.end()) {
      return it->second->createNode();
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

#define REGISTER_NODE(node_name, node_class)               \
  static auto node_creator_##node_name = []() {            \
    NodeFactory::getInstance()->registerNode(              \
        node_name, new NodeCreatorRegister<node_class>()); \
    return 0;                                              \
  }();

using SISONodeFunc =
    std::function<base::Status(Edge *input, Edge *output, base::Param *param)>;

using SIMONodeFunc = std::function<base::Status(
    Edge *input, std::initializer_list<Edge *> outputs, base::Param *param)>;

using MISONodeFunc = std::function<base::Status(
    std::initializer_list<Edge *> inputs, Edge *output, base::Param *param)>;

using MIMONodeFunc = std::function<base::Status(
    std::initializer_list<Edge *> inputs, std::initializer_list<Edge *> outputs,
    base::Param *param)>;

}  // namespace dag
}  // namespace nndeploy

#endif
