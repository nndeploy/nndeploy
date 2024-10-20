
#include "nndeploy/dag/graph.h"

#include "nndeploy/base/common.h"
#include "nndeploy/base/glic_stl_include.h"
#include "nndeploy/base/log.h"
#include "nndeploy/base/macro.h"
#include "nndeploy/base/object.h"
#include "nndeploy/base/status.h"
#include "nndeploy/base/string.h"
#include "nndeploy/base/time_profiler.h"
#include "nndeploy/base/value.h"
#include "nndeploy/dag/edge.h"
#include "nndeploy/dag/executor/parallel_pipeline_executor.h"
#include "nndeploy/dag/executor/parallel_task_executor.h"
#include "nndeploy/dag/executor/sequential_executor.h"
#include "nndeploy/dag/node.h"
#include "nndeploy/dag/util.h"
#include "nndeploy/device/buffer.h"
#include "nndeploy/device/device.h"
#include "nndeploy/device/memory_pool.h"
#include "nndeploy/device/tensor.h"

namespace nndeploy {
namespace dag {

Graph::Graph(const std::string &name, Edge *input, Edge *output)
    : Node(name, input, output) {
  if (input != nullptr) {
    if (nullptr == addEdge(input)) {
      constructed_ = false;
      return;
    }
  }
  if (output != nullptr) {
    if (nullptr == addEdge(output)) {
      constructed_ = false;
      return;
    }
  }
  constructed_ = true;
}
Graph::Graph(const std::string &name, std::initializer_list<Edge *> inputs,
             std::initializer_list<Edge *> outputs)
    : Node(name, inputs, outputs) {
  for (auto input : inputs) {
    if (nullptr == addEdge(input)) {
      constructed_ = false;
      return;
    }
  }
  for (auto output : outputs) {
    if (nullptr == addEdge(output)) {
      constructed_ = false;
      return;
    }
  }
  constructed_ = true;
}
Graph::Graph(const std::string &name, std::vector<Edge *> inputs,
             std::vector<Edge *> outputs)
    : Node(name, inputs, outputs) {
  for (auto input : inputs) {
    if (nullptr == addEdge(input)) {
      constructed_ = false;
      return;
    }
  }
  for (auto output : outputs) {
    if (nullptr == addEdge(output)) {
      constructed_ = false;
      return;
    }
  }
  constructed_ = true;
}
Graph::~Graph() {
  for (auto node_wrapper : node_repository_) {
    if (!node_wrapper->is_external_) {
      delete node_wrapper->node_;
    }
    delete node_wrapper;
  }
  for (auto edge_wrapper : edge_repository_) {
    if (!edge_wrapper->is_external_) {
      delete edge_wrapper->edge_;
    }
    delete edge_wrapper;
  }
  node_repository_.clear();
  edge_repository_.clear();
}

Edge *Graph::createEdge(const std::string &name) {
  Edge *edge = new Edge(name);
  EdgeWrapper *edge_wrapper = new EdgeWrapper();
  edge_wrapper->is_external_ = false;
  edge_wrapper->edge_ = edge;
  edge_wrapper->name_ = name;
  edge_repository_.emplace_back(edge_wrapper);
  return edge;
}

Edge *Graph::getEdge(const std::string &name) {
  for (EdgeWrapper *edge_wrapper : edge_repository_) {
    if (edge_wrapper->name_ == name) {
      return edge_wrapper->edge_;
    }
  }
  return nullptr;
}

// EdgeWrapper *Graph::addEdge(Edge *edge) {
//   base::Status status = base::kStatusCodeOk;
//   NNDEPLOY_CHECK_PARAM_NULL_RET_NULL(edge, "edge is null!");
//   EdgeWrapper *edge_wrapper = new EdgeWrapper();
//   edge_wrapper->is_external_ = true;
//   edge_wrapper->edge_ = edge;
//   edge_wrapper->name_ = edge->getName();
//   edge_repository_.emplace_back(edge_wrapper);
//   return edge_wrapper;
// }

EdgeWrapper *Graph::addEdge(Edge *edge, bool is_external) {
  base::Status status = base::kStatusCodeOk;
  NNDEPLOY_CHECK_PARAM_NULL_RET_NULL(edge, "edge is null!");
  EdgeWrapper *edge_wrapper = new EdgeWrapper();
  edge_wrapper->is_external_ = is_external;
  edge_wrapper->edge_ = edge;
  edge_wrapper->name_ = edge->getName();
  edge_repository_.emplace_back(edge_wrapper);
  return edge_wrapper;
}

// base::Status Graph::addNode(Node *node) {
//   base::Status status = base::kStatusCodeOk;
//   NNDEPLOY_CHECK_PARAM_NULL_RET_STATUS(node, "node is null!");
//   NodeWrapper *node_wrapper = new NodeWrapper();
//   node_wrapper->is_external_ = true;
//   node_wrapper->node_ = node;
//   node_wrapper->name_ = node->getName();
//   for (auto input : node->getAllInput()) {
//     EdgeWrapper *input_wrapper = findEdgeWrapper(edge_repository_, input);
//     if (input_wrapper == nullptr) {
//       input_wrapper = this->addEdge(input);
//     }
//     input_wrapper->consumers_.emplace_back(node_wrapper);
//   }
//   for (auto output : node->getAllOutput()) {
//     EdgeWrapper *output_wrapper = findEdgeWrapper(edge_repository_, output);
//     if (output_wrapper == nullptr) {
//       output_wrapper = this->addEdge(output);
//     }
//     output_wrapper->producers_.emplace_back(node_wrapper);
//   }

//   node_repository_.emplace_back(node_wrapper);
//   return status;
// }

base::Status Graph::addNode(Node *node, bool is_external) {
  base::Status status = base::kStatusCodeOk;
  NNDEPLOY_CHECK_PARAM_NULL_RET_STATUS(node, "node is null!");
  NodeWrapper *node_wrapper = new NodeWrapper();
  node_wrapper->is_external_ = is_external;
  node_wrapper->node_ = node;
  node_wrapper->name_ = node->getName();
  for (auto input : node->getAllInput()) {
    EdgeWrapper *input_wrapper = findEdgeWrapper(edge_repository_, input);
    if (input_wrapper == nullptr) {
      input_wrapper = this->addEdge(input, is_external);
    }
    // input_wrapper->consumers_.emplace_back(node_wrapper);
    insertUnique(input_wrapper->consumers_, node_wrapper);
  }
  for (auto output : node->getAllOutput()) {
    EdgeWrapper *output_wrapper = findEdgeWrapper(edge_repository_, output);
    if (output_wrapper == nullptr) {
      output_wrapper = this->addEdge(output, is_external);
    }
    // output_wrapper->producers_.emplace_back(node_wrapper);
    insertUnique(output_wrapper->producers_, node_wrapper);
  }

  node_repository_.emplace_back(node_wrapper);
  return status;
}

base::Status Graph::setNodeParam(const std::string &node_name,
                                 base::Param *param) {
  base::Status status = base::kStatusCodeOk;
  NNDEPLOY_CHECK_PARAM_NULL_RET_STATUS(param, "param is null!");
  NodeWrapper *node_wrapper = findNodeWrapper(node_repository_, node_name);
  NNDEPLOY_CHECK_PARAM_NULL_RET_STATUS(node_wrapper, "node_wrapper is null!");
  status = node_wrapper->node_->setParam(param);
  return status;
}

base::Param *Graph::getNodeParam(const std::string &node_name) {
  NodeWrapper *node_wrapper = findNodeWrapper(node_repository_, node_name);
  NNDEPLOY_CHECK_PARAM_NULL_RET_NULL(node_wrapper, "node_wrapper is null!");
  return node_wrapper->node_->getParam();
}

base::Status Graph::init() {
  base::Status status = base::kStatusCodeOk;

  // NNDEPLOY_LOGI("###########################\n");
  // NNDEPLOY_LOGI("setInitializedFlag false!\n");
  // NNDEPLOY_LOGI("###########################\n");
  setInitializedFlag(false);

  status = this->construct();
  NNDEPLOY_RETURN_ON_NEQ(status, base::kStatusCodeOk,
                         "graph construct failed!");

  status = this->executor();
  NNDEPLOY_RETURN_ON_NEQ(status, base::kStatusCodeOk, "graph executor failed!");

  // NNDEPLOY_LOGI("###########################\n");
  // NNDEPLOY_LOGI("setInitializedFlag true!\n");
  // NNDEPLOY_LOGI("###########################\n");
  setInitializedFlag(true);

  return status;
}

base::Status Graph::deinit() {
  base::Status status = base::kStatusCodeOk;

  // NNDEPLOY_LOGI("###########################\n");
  // NNDEPLOY_LOGI("setInitializedFlag false!\n");
  // NNDEPLOY_LOGI("###########################\n");
  setInitializedFlag(false);

  // NNDEPLOY_LOGI("#######################\n");
  // NNDEPLOY_LOGI("Node DeInitialize Phase!\n");
  // NNDEPLOY_LOGI("#######################\n");
  status = executor_->deinit();
  NNDEPLOY_RETURN_ON_NEQ(status, base::kStatusCodeOk,
                         "executor deinit failed!");
  return status;
}

base::Status Graph::run() {
  base::Status status = base::kStatusCodeOk;

  // NNDEPLOY_LOGI("###########################\n");
  // NNDEPLOY_LOGI("setRunningFlag true!\n");
  // NNDEPLOY_LOGI("###########################\n");
  setRunningFlag(true);

  // NNDEPLOY_LOGI("#######################\n");
  // NNDEPLOY_LOGI("Node run Phase!\n");
  // NNDEPLOY_LOGI("#######################\n");
  status = executor_->run();
  NNDEPLOY_RETURN_ON_NEQ(status, base::kStatusCodeOk, "executor run failed!");

  // NNDEPLOY_LOGI("###########################\n");
  // NNDEPLOY_LOGI("setRunningFlag false!\n");
  // NNDEPLOY_LOGI("###########################\n");
  setRunningFlag(false);

  return status;
}

base::Status Graph::dump(std::ostream &oss) {
  base::Status status = dumpDag(edge_repository_, node_repository_, inputs_,
                                outputs_, name_, oss);
  NNDEPLOY_RETURN_ON_NEQ(status, base::kStatusCodeOk, "dump failed!");
  return status;
}

base::Status Graph::construct() {
  base::Status status = base::kStatusCodeOk;

  NNDEPLOY_LOGE("NAME: %s start\n", name_.c_str());

  // NNDEPLOY_LOGI("###########################\n");
  // NNDEPLOY_LOGI("parallel_type_!\n");
  // NNDEPLOY_LOGI("###########################\n");
  // base::ParallelType parallel_type_ = parallel_type_;

  // NNDEPLOY_LOGI("###########################\n");
  // NNDEPLOY_LOGI("Parameter Validation Phase!\n");
  // NNDEPLOY_LOGI("###########################\n");
  for (auto node_wrapper : node_repository_) {
    NNDEPLOY_CHECK_PARAM_NULL_RET_STATUS(node_wrapper->node_,
                                         "edge_repository_ node is null!");
  }
  for (auto edge_wrapper : edge_repository_) {
    NNDEPLOY_CHECK_PARAM_NULL_RET_STATUS(edge_wrapper->edge_,
                                         "edge_repository_ edge is null!");
    if (edge_wrapper->producers_.empty() && edge_wrapper->consumers_.empty()) {
      NNDEPLOY_LOGI("this edge[%s] is unuseless!\n",
                    edge_wrapper->edge_->getName().c_str());
    }
  }

  // NNDEPLOY_LOGI("####################\n");
  // NNDEPLOY_LOGI("Mark Predecessors And Successors Phase!\n");
  // NNDEPLOY_LOGI("####################\n");
  for (auto node_wrapper : node_repository_) {
    Node *node = node_wrapper->node_;
    node->setDebugFlag(is_debug_);
    node->setTimeProfileFlag(is_time_profile_);
    node->setParallelType(parallel_type_);
    node->setInnerFlag(true);
    std::vector<Edge *> inputs = node->getAllInput();
    for (auto input : inputs) {
      EdgeWrapper *input_wrapper = findEdgeWrapper(edge_repository_, input);
      NNDEPLOY_CHECK_PARAM_NULL_RET_STATUS(input_wrapper,
                                           "input_wrapper is null!");

      for (auto producer : input_wrapper->producers_) {
        insertUnique(node_wrapper->predecessors_, producer);
      }
    }
    std::vector<Edge *> outputs = node->getAllOutput();
    for (auto output : outputs) {
      EdgeWrapper *output_wrapper = findEdgeWrapper(edge_repository_, output);
      NNDEPLOY_CHECK_PARAM_NULL_RET_STATUS(output_wrapper,
                                           "output_wrapper is null!");

      for (auto consumer : output_wrapper->consumers_) {
        insertUnique(node_wrapper->successors_, consumer);
      }
    }
  }

  // NNDEPLOY_LOGI("##############\n");
  // NNDEPLOY_LOGI("construct edge\n");
  // NNDEPLOY_LOGI("##############\n");
  for (auto edge_wrapper : edge_repository_) {
    std::vector<Node *> producers;
    for (auto producer : edge_wrapper->producers_) {
      producers.emplace_back(producer->node_);
    }
    std::vector<Node *> consumers;
    for (auto consumer : edge_wrapper->consumers_) {
      consumers.emplace_back(consumer->node_);
    }
    base::Status status = edge_wrapper->edge_->setParallelType(parallel_type_);
    NNDEPLOY_RETURN_ON_NEQ(status, base::kStatusCodeOk,
                           "setParallelType failed!");
    // 必须在abstract_edge管理该字段
    status = edge_wrapper->edge_->increaseProducers(producers);
    NNDEPLOY_RETURN_ON_NEQ(status, base::kStatusCodeOk,
                           "increaseProducers failed!");
    status = edge_wrapper->edge_->increaseConsumers(consumers);
    NNDEPLOY_RETURN_ON_NEQ(status, base::kStatusCodeOk,
                           "increaseConsumers failed!");
    status = edge_wrapper->edge_->construct();
    NNDEPLOY_RETURN_ON_NEQ(status, base::kStatusCodeOk,
                           "construct edge failed!");
  }

  if (!is_inner_) {
    for (auto iter : outputs_) {
      iter->markGraphOutput();
    }
  }

  NNDEPLOY_LOGE("NAME: %s end\n", name_.c_str());

  return status;
}

base::Status Graph::executor() {
  // NNDEPLOY_LOGI("name: %s executor start.\n", name_.c_str());
  base::Status status = base::kStatusCodeOk;

  // NNDEPLOY_LOGI("###########################\n");
  // NNDEPLOY_LOGI("parallel_type_!\n");
  // NNDEPLOY_LOGI("###########################\n");
  // base::ParallelType parallel_type_ = parallel_type_;

  // NNDEPLOY_LOGI("##############\n");
  // NNDEPLOY_LOGI("create executor\n");
  // NNDEPLOY_LOGI("##############\n");
  if (parallel_type_ == base::kParallelTypeNone) {
    executor_ = std::make_shared<SequentialExecutor>();
  } else if (parallel_type_ == base::kParallelTypeSequential) {
    executor_ = std::make_shared<SequentialExecutor>();
  } else if (parallel_type_ == base::kParallelTypeTask) {
    executor_ = std::make_shared<ParallelTaskExecutor>();
  } else if (parallel_type_ == base::kParallelTypePipeline) {
    executor_ = std::make_shared<ParallelPipelineExecutor>();
  } else {
    NNDEPLOY_LOGE("parallel_type_ is invalid!\n");
    return base::kStatusCodeErrorInvalidValue;
  }
  NNDEPLOY_CHECK_PARAM_NULL_RET_STATUS(executor_, "Create executor failed!");

  // NNDEPLOY_LOGI("##############\n");
  // NNDEPLOY_LOGI("executor init\n");
  // NNDEPLOY_LOGI("1. Optimizer Graph V1!\n");
  // NNDEPLOY_LOGI("2. Device Verification Phase!\n");
  // NNDEPLOY_LOGI("3. Optimizer Graph V2!\n");
  // NNDEPLOY_LOGI("4. Memory Allocation Phase!\n");
  // NNDEPLOY_LOGI("5. Cost Calculations!\n");
  // NNDEPLOY_LOGI("##############\n");
  status = executor_->init(edge_repository_, node_repository_);
  NNDEPLOY_RETURN_ON_NEQ(status, base::kStatusCodeOk, "executor init failed!");

  // NNDEPLOY_LOGI("name: %s executor start.\n", name_.c_str());
  return status;
}

std::map<std::string, createGraphFunc> &getGlobalGraphCreatorMap() {
  static std::once_flag once;
  static std::shared_ptr<std::map<std::string, createGraphFunc>> creators;
  std::call_once(once, []() {
    creators.reset(new std::map<std::string, createGraphFunc>);
  });
  return *creators;
}

Graph *createGraph(const std::string &name, base::InferenceType inference_type,
                   base::DeviceType device_type, Edge *input, Edge *output,
                   base::ModelType model_type, bool is_path,
                   std::vector<std::string> model_value) {
  Graph *temp = nullptr;
  auto &creater_map = getGlobalGraphCreatorMap();
  if (creater_map.count(name) > 0) {
    temp = creater_map[name](name, inference_type, device_type, input, output,
                             model_type, is_path, model_value);
  }
  return temp;
}

}  // namespace dag
}  // namespace nndeploy
