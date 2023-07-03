
#include "nndeploy/task/pipeline.h"

#include "nndeploy/base/basic.h"
#include "nndeploy/base/glic_stl_include.h"
#include "nndeploy/base/log.h"
#include "nndeploy/base/macro.h"
#include "nndeploy/base/object.h"
#include "nndeploy/base/status.h"
#include "nndeploy/base/string.h"
#include "nndeploy/base/value.h"
#include "nndeploy/device/buffer.h"
#include "nndeploy/device/buffer_pool.h"
#include "nndeploy/device/device.h"
#include "nndeploy/device/tensor.h"
#include "nndeploy/inference/abstract_inference.h"
#include "nndeploy/inference/inference_param.h"
#include "nndeploy/task/packet.h"
#include "nndeploy/task/task.h"

namespace nndeploy {
namespace task {

enum TaskColorType : int32_t {
  kTaskColorWhite = 0x0000,
  kTaskColorGray,
  kTaskColorBlack
};

class TaskWrapper {
 public:
  bool is_external_;
  Task* task_;
  std::string name_;
  std::vector<TaskWrapper*> predecessors_;
  std::vector<TaskWrapper*> successors_;
  TaskColorType color_ = kTaskColorWhite;
};

class PacketWrapper {
 public:
  bool is_external_;
  Packet* packet_;
  std::vector<TaskWrapper*> producers_;
  std::vector<TaskWrapper*> consumers_;
};

Pipeline::Pipeline(const std::string& name, Packet* input, Packet* output)
    : Task(name, input, output) {
  param_ = std::make_shared<PipelineParam>();
  base::Status status = base::kStatusCodeOk;
  status = addPacket(input);
  if (status != base::kStatusCodeOk) {
    constructed_ = false;
    return;
  }
  status = addPacket(output);
  if (status != base::kStatusCodeOk) {
    constructed_ = false;
    return;
  }
  constructed_ = true;
}
Pipeline::Pipeline(const std::string& name, std::vector<Packet*> inputs,
                   std::vector<Packet*> outputs)
    : Task(name, inputs, outputs) {
  param_ = std::make_shared<PipelineParam>();
  base::Status status = base::kStatusCodeOk;
  for (auto input : inputs) {
    status = addPacket(input);
    if (status != base::kStatusCodeOk) {
      constructed_ = false;
      return;
    }
  }
  for (auto output : outputs) {
    status = addPacket(output);
    if (status != base::kStatusCodeOk) {
      constructed_ = false;
      return;
    }
  }
  constructed_ = true;
}
Pipeline::~Pipeline() {
  for (auto task_wrapper : task_repository_) {
    if (!task_wrapper->is_external_) {
      delete task_wrapper->task_;
    }
    delete task_wrapper;
  }
  for (auto packet_wrapper : packet_repository_) {
    if (!packet_wrapper->is_external_) {
      delete packet_wrapper->packet_;
    }
    delete packet_wrapper;
  }
  topo_sort_task_.clear();
  task_repository_.clear();
  packet_repository_.clear();
}

Packet* Pipeline::createPacket(const std::string& name) {
  Packet* packet = new Packet(name);
  PacketWrapper* packet_wrapper = new PacketWrapper();
  packet_wrapper->is_external_ = false;
  packet_wrapper->packet_ = packet;
  packet_repository_.emplace_back(packet_wrapper);
  return packet;
}
base::Status Pipeline::addPacket(Packet* packet) {
  base::Status status = base::kStatusCodeOk;
  NNDEPLOY_CHECK_PARAM_NULL(packet);
  PacketWrapper* packet_wrapper = new PacketWrapper();
  packet_wrapper->is_external_ = true;
  packet_wrapper->packet_ = packet;
  packet_repository_.emplace_back(packet_wrapper);
  return status;
}

template <typename T>
Task* Pipeline::createTask(const std::string& name, Packet* input,
                           Packet* output) {
  NNDEPLOY_CHECK_PARAM_NULL(input);
  NNDEPLOY_CHECK_PARAM_NULL(output);
  Task* task = dynamic_cast<Task*>(new T(name, input, output));
  TaskWrapper* task_wrapper = new TaskWrapper();
  task_wrapper->is_external_ = false;
  task_wrapper->task_ = task;
  task_wrapper->name_ = name;
  if (findPacketWrapper(input) == nullptr) {
    this->addPacket(input);
  }
  findPacketWrapper(input)->consumers_.emplace_back(task_wrapper);
  if (findPacketWrapper(output) == nullptr) {
    this->addPacket(output);
  }
  findPacketWrapper(output)->producers_.emplace_back(task_wrapper);
  task_repository_.emplace_back(task_wrapper);
  return task;
}
template <typename T>
Task* Pipeline::createTask(const std::string& name, std::vector<Packet*> inputs,
                           std::vector<Packet*> outputs) {
  if (inputs.empty() || outputs.empty()) {
    NNDEPLOY_LOGE("inputs or outputs is empty!\n")
    return nullptr;
  }
  Task* task = dynamic_cast<Task*>(new T(name, inputs, outputs));
  TaskWrapper* task_wrapper = new TaskWrapper();
  task_wrapper->is_external_ = false;
  task_wrapper->task_ = task;
  task_wrapper->name_ = name;
  for (auto input : inputs) {
    if (findPacketWrapper(input) == nullptr) {
      this->addPacket(input);
    }
    findPacketWrapper(input)->consumers_.emplace_back(task_wrapper);
  }
  for (auto output : outputs) {
    if (findPacketWrapper(output) == nullptr) {
      this->addPacket(output);
    }
    findPacketWrapper(output)->producers_.emplace_back(task_wrapper);
  }
  task_repository_.emplace_back(task_wrapper);
  return task;
}
template <typename T>
Task* Pipeline::createInference(const std::string& name,
                                base::InferenceType type, Packet* input,
                                Packet* output) {
  NNDEPLOY_CHECK_PARAM_NULL(input);
  NNDEPLOY_CHECK_PARAM_NULL(output);
  Task* task = dynamic_cast<Task*>(new T(name, type, input, output));
  TaskWrapper* task_wrapper = new TaskWrapper();
  task_wrapper->is_external_ = false;
  task_wrapper->task_ = task;
  task_wrapper->name_ = name;
  if (findPacketWrapper(input) == nullptr) {
    this->addPacket(input);
  }
  findPacketWrapper(input)->consumers_.emplace_back(task_wrapper);
  if (findPacketWrapper(output) == nullptr) {
    this->addPacket(output);
  }
  findPacketWrapper(output)->producers_.emplace_back(task_wrapper);
  task_repository_.emplace_back(task_wrapper);
  return task;
}
base::Status Pipeline::addTask(Task* task) {
  base::Status status = base::kStatusCodeOk;
  NNDEPLOY_CHECK_PARAM_NULL(task);
  TaskWrapper* task_wrapper = new TaskWrapper();
  task_wrapper->is_external_ = true;
  task_wrapper->task_ = task;
  task_wrapper->name_ = task->getName();
  for (auto input : task->getAllInput()) {
    if (findPacketWrapper(input) == nullptr) {
      this->addPacket(input);
    }
    findPacketWrapper(input)->consumers_.emplace_back(task_wrapper);
  }
  for (auto output : task->getAllOutput()) {
    if (findPacketWrapper(output) == nullptr) {
      this->addPacket(output);
    }
    findPacketWrapper(output)->producers_.emplace_back(task_wrapper);
  }
  task_repository_.emplace_back(task_wrapper);
  return status;
}

base::Status Pipeline::init() {
  base::Status status = base::kStatusCodeOk;

  NNDEPLOY_LOGI("###########################\n");
  NNDEPLOY_LOGI("Parameter Validation Phase!\n");
  NNDEPLOY_LOGI("###########################\n");
  for (auto task_wrapper : task_repository_) {
    NNDEPLOY_CHECK_PARAM_NULL(task_wrapper->task_);
  }
  for (auto packet_wrapper : packet_repository_) {
    NNDEPLOY_CHECK_PARAM_NULL(packet_wrapper->packet_);
  }

  NNDEPLOY_LOGI("####################\n");
  NNDEPLOY_LOGI("Mark Predecessors And Successors Phase!\n");
  NNDEPLOY_LOGI("####################\n");
  for (auto task_wrapper : task_repository_) {
    Task* task = task_wrapper->task_;
    std::vector<Packet*> inputs = task->getAllInput();
    for (auto input : inputs) {
      PacketWrapper* input_wrapper = findPacketWrapper(input);
      NNDEPLOY_CHECK_PARAM_NULL(input_wrapper);
      task_wrapper->predecessors_.assign(input_wrapper->producers_.begin(),
                                         input_wrapper->producers_.end());
    }
    std::vector<Packet*> outputs = task->getAllOutput();
    for (auto output : outputs) {
      PacketWrapper* output_wrapper = findPacketWrapper(output);
      NNDEPLOY_CHECK_PARAM_NULL(output_wrapper);
      task_wrapper->successors_.assign(output_wrapper->consumers_.begin(),
                                       output_wrapper->consumers_.end());
    }
  }

  NNDEPLOY_LOGI("##############\n");
  NNDEPLOY_LOGI("TopologicalSort and Check Cycle!\n");
  NNDEPLOY_LOGI("##############\n");
  status = topologicalSort();
  if (status != base::kStatusCodeOk) {
    NNDEPLOY_LOGE("Toposort failed");
    return status;
  }

  // NNDEPLOY_LOGI("############################\n");
  // NNDEPLOY_LOGI("Checking for Unvisited Packet!\n");
  // NNDEPLOY_LOGI("############################\n");

  // NNDEPLOY_LOGI("############################\n");
  // NNDEPLOY_LOGI("Optimizer Graph V1!\n");
  // NNDEPLOY_LOGI("############################\n");

  // NNDEPLOY_LOGI("#########################\n");
  // NNDEPLOY_LOGI("Device Verification Phase!\n");
  // NNDEPLOY_LOGI("#########################\n");

  // NNDEPLOY_LOGI("############################\n");
  // NNDEPLOY_LOGI("Optimizer Graph V2!\n");
  // NNDEPLOY_LOGI("############################\n");

  NNDEPLOY_LOGI("#######################\n");
  NNDEPLOY_LOGI("Task Initialize Phase!\n");
  NNDEPLOY_LOGI("#######################\n");
  for (auto task_vec : topo_sort_task_) {
    for (auto task : task_vec) {
      status = task->init();
      if (status != base::kStatusCodeOk) {
        NNDEPLOY_LOGE("Task init failed!\n");
        return status;
      }
    }
  }

  // NNDEPLOY_LOGI("########################\n");
  // NNDEPLOY_LOGI("Memory Allocation Phase!\n");
  // NNDEPLOY_LOGI("########################\n");

  // NNDEPLOY_LOGI("#######################\n");
  // NNDEPLOY_LOGI("Cost Calculations!\n");
  // NNDEPLOY_LOGI("#######################\n");

  return status;
}

base::Status Pipeline::deinit() {
  base::Status status = base::kStatusCodeOk;
  NNDEPLOY_LOGI("#######################\n");
  NNDEPLOY_LOGI("Task DeInitialize Phase!\n");
  NNDEPLOY_LOGI("#######################\n");
  for (auto task_vec : topo_sort_task_) {
    for (auto task : task_vec) {
      status = task->deinit();
      if (status != base::kStatusCodeOk) {
        NNDEPLOY_LOGE("Task deinit failed!\n");
        return status;
      }
    }
  }
  return status;
}

base::Status Pipeline::reShape() {
  base::Status status = base::kStatusCodeOk;
  NNDEPLOY_LOGI("#######################\n");
  NNDEPLOY_LOGI("Task reShape Phase!\n");
  NNDEPLOY_LOGI("#######################\n");
  for (auto task_vec : topo_sort_task_) {
    for (auto task : task_vec) {
      status = task->reShape();
      if (status != base::kStatusCodeOk) {
        NNDEPLOY_LOGE("Task run failed!\n");
        return status;
      }
    }
  }
  return status;
}

base::Status Pipeline::run() {
  base::Status status = base::kStatusCodeOk;
  NNDEPLOY_LOGI("#######################\n");
  NNDEPLOY_LOGI("Task DeInitialize Phase!\n");
  NNDEPLOY_LOGI("#######################\n");
  for (auto task_vec : topo_sort_task_) {
    for (auto task : task_vec) {
      status = task->run();
      if (status != base::kStatusCodeOk) {
        NNDEPLOY_LOGE("Task run failed!\n");
        return status;
      }
    }
  }
  return status;
}

// base::Status Pipeline::dump(std::ostream& oss = std::cout);

PacketWrapper* Pipeline::findPacketWrapper(Packet* packet) {
  for (auto packet_wrapper : packet_repository_) {
    if (packet_wrapper->packet_ == packet) {
      return packet_wrapper;
    }
  }
  return nullptr;
}
TaskWrapper* Pipeline::findTaskWrapper(Task* task) {
  for (auto task_wrapper : task_repository_) {
    if (task_wrapper->task_ == task) {
      return task_wrapper;
    }
  }
  return nullptr;
}

std::vector<TaskWrapper*> Pipeline::findStartTasks() {
  std::vector<TaskWrapper*> start_tasks;
  for (auto task_wrapper : task_repository_) {
    if (task_wrapper->predecessors_.empty()) {
      start_tasks.emplace_back(task_wrapper);
    }
  }
  return start_tasks;
}

std::vector<TaskWrapper*> Pipeline::findEndTasks() {
  std::vector<TaskWrapper*> end_tasks;
  for (auto task_wrapper : task_repository_) {
    if (task_wrapper->successors_.empty()) {
      end_tasks.emplace_back(task_wrapper);
    }
  }
  return end_tasks;
}

base::Status Pipeline::TopoSortBFS(TaskWrapper* task_wrapper) {
  std::vector<Task*> dst;
  task_wrapper->color_ = kTaskColorGray;
  std::deque<TaskWrapper*> task_deque;
  task_deque.emplace_back(task_wrapper);
  while (!task_deque.empty()) {
    TaskWrapper* task_wrapper = task_deque.front();
    if (task_wrapper->color_ == kTaskColorBlack) {
      task_deque.pop_front();
      continue;
    }
    bool flag = false;
    for (auto predecessor : task_wrapper->predecessors_) {
      if (predecessor->color_ != kTaskColorBlack) {
        predecessor->color_ = kTaskColorGray;
        task_deque.emplace_front(predecessor);
        flag = true;
        break;
      }
    }
    if (flag) {
      continue;
    }
    for (auto successor : task_wrapper->successors_) {
      if (successor->color_ == kTaskColorBlack) {
        NNDEPLOY_LOGE("Cycle detected in pipeline");
        return base::kStatusCodeErrorInvalidValue;
      } else if (successor->color_ == kTaskColorWhite) {
        successor->color_ = kTaskColorGray;
        task_deque.emplace_back(successor);
      }
    }
    task_deque.pop_front();
    task_wrapper->color_ = kTaskColorBlack;
    dst.emplace_back(task_wrapper->task_);
  }
  topo_sort_task_.push_back(dst);
  return base::kStatusCodeOk;
}

base::Status Pipeline::TopoSortDFS(TaskWrapper* task_wrapper,
                                   std::stack<TaskWrapper*>& dst) {
  base::Status status = base::kStatusCodeOk;
  task_wrapper->color_ = kTaskColorGray;
  for (auto successor : task_wrapper->successors_) {
    if (successor->color_ == kTaskColorWhite) {
      status = TopoSortDFS(successor, dst);
    } else if (successor->color_ == kTaskColorGray) {
      NNDEPLOY_LOGE("Cycle detected in pipeline");
      status = base::kStatusCodeErrorInvalidValue;
    } else {
      continue;
    }
  }
  if (status != base::kStatusCodeOk) {
    return status;
  }
  task_wrapper->color_ = kTaskColorBlack;
  dst.push(task_wrapper);
  return base::kStatusCodeOk;
}

/**
 * @brief topo sort and check cycle
 *
 * @return base::Status
 */
base::Status Pipeline::topologicalSort() {
  base::Status status = base::kStatusCodeOk;

  std::vector<TaskWrapper*> start_tasks = findStartTasks();
  if (start_tasks.empty()) {
    NNDEPLOY_LOGE("No start task found in pipeline");
    return base::kStatusCodeErrorInvalidValue;
  }
  PipelineParam* param = dynamic_cast<PipelineParam*>(this->param_.get());
  if (param->topo_sort_type_ == kTopoSortTypeBFS) {
    for (auto task_wrapper : start_tasks) {
      if (task_wrapper->color_ == kTaskColorBlack) {
        continue;
      }
      status = TopoSortBFS(task_wrapper);
      if (status != base::kStatusCodeOk) {
        NNDEPLOY_LOGE("TopoSortBFS failed");
        return status;
      }
    }
  } else {
    std::stack<TaskWrapper*> dst;
    for (auto task_wrapper : start_tasks) {
      if (task_wrapper->color_ == kTaskColorBlack) {
        continue;
      }
      status = TopoSortDFS(task_wrapper, dst);
      if (status != base::kStatusCodeOk) {
        NNDEPLOY_LOGE("TopoSortDFS failed");
        return status;
      }
    }
    std::vector<Task*> task_dst;
    while (!dst.empty()) {
      task_dst.emplace_back(dst.top()->task_);
      dst.pop();
    }
    topo_sort_task_.push_back(task_dst);
  }

  return status;
}

}  // namespace task
}  // namespace nndeploy
