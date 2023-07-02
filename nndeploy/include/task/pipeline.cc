#ifndef _NNDEPLOY_SOURCE_TASK_PIPELINE_H_
#define _NNDEPLOY_SOURCE_TASK_PIPELINE_H_

#include "nndeploy/source/base/basic.h"
#include "nndeploy/source/base/glic_stl_include.h"
#include "nndeploy/source/base/log.h"
#include "nndeploy/source/base/macro.h"
#include "nndeploy/source/base/object.h"
#include "nndeploy/source/base/status.h"
#include "nndeploy/source/base/string.h"
#include "nndeploy/source/base/value.h"
#include "nndeploy/source/device/buffer.h"
#include "nndeploy/source/device/buffer_pool.h"
#include "nndeploy/source/device/device.h"
#include "nndeploy/source/device/tensor.h"
#include "nndeploy/source/inference/inference.h"
#include "nndeploy/source/inference/inference_param.h"
#include "nndeploy/source/task/detr_cpu_pipeline.h"
#include "nndeploy/source/task/packet.h"
#include "nndeploy/source/task/task.h"

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
  TaskColorType TaskColorType = kTaskColorWhite;
}

class PacketWrapper {
 public:
  bool is_external_;
  Packet* packet_;
  std::vector<TaskWrapper*> producers_;
  std::vector<TaskWrapper*> consumers_;
}

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
~Pipeline::Pipeline() {
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
void Pipeline::addTask(Task* task) {
  NNDEPLOY_CHECK_PARAM_NULL(task);
  TaskWrapper* task_wrapper = new TaskWrapper();
  task_wrapper->is_external_ = true;
  task_wrapper->task_ = task;
  task_wrapper->name_ = task->getName();
  for (auto input : task_->getAllInput()) {
    if (findPacketWrapper(input) == nullptr) {
      this->addPacket(input);
    }
    findPacketWrapper(input)->consumers_.emplace_back(task_wrapper);
  }
  for (auto output : task_->getAllOutput()) {
    if (findPacketWrapper(output) == nullptr) {
      this->addPacket(output);
    }
    findPacketWrapper(output)->producers_.emplace_back(task_wrapper);
  }
  task_repository_.emplace_back(task_wrapper);
}

base::Status Pipeline::init() {
  base::Status status = base::kStatusCodeOK;

  NNDEPLOY_LOGI("###########################\n");
  NNDEPLOY_LOGI("Parameter Validation Phase!\n");
  NNDEPLOY_LOGI("###########################\n");
  for (auto task_wrapper : task_repository_) {
    NNDEPLOY_CHECK_PARAM_NULL(task_wrapper->task_)
  }
  for (auto packet_wrapper : packet_repository_) {
    NNDEPLOY_CHECK_PARAM_NULL(packet_wrapper->packet_)
  }

  NNDEPLOY_LOGI(VX_ZONE_GRAPH, "####################\n");
  NNDEPLOY_LOGI(VX_ZONE_GRAPH, "Mark Predecessors And Successors Phase!\n");
  NNDEPLOY_LOGI(VX_ZONE_GRAPH, "####################\n");
  for (auto task_wrapper : task_repository_) {
    Task* task = task_wrapper->task_;
    std::vector<Packet*> inputs = task->getAllInput();
    for (auto input : inputs) {
      PacketWrapper* input_wrapper = findPacketWrapper(input);
      NNDEPLOY_CHECK_PARAM_NULL(input_wrapper);
      task_wrapper->predecessors_.assign(input->producers_.begin(),
                                         input->producers_.end());
    }
    std::vector<Packet*> outputs = task->getAllOutput();
    for (auto output : outputs) {
      PacketWrapper* output_wrapper = findPacketWrapper(output);
      NNDEPLOY_CHECK_PARAM_NULL(output_wrapper);
      task_wrapper->successors_.assign(output->consumers_.begin(),
                                       output->consumers_.end());
    }
  }

  NNDEPLOY_LOGI(VX_ZONE_GRAPH, "##############\n");
  NNDEPLOY_LOGI(VX_ZONE_GRAPH, "TopologicalSort and Check Cycle!\n");
  NNDEPLOY_LOGI(VX_ZONE_GRAPH, "##############\n");
  status = TopologicalSort();
  if (status != base::kStatusCodeOK) {
    NNDEPLOY_LOG_ERROR("Toposort failed");
    return status;
  }

  // NNDEPLOY_LOGI(VX_ZONE_GRAPH, "############################\n");
  // NNDEPLOY_LOGI(VX_ZONE_GRAPH, "Checking for Unvisited Packet!\n");
  // NNDEPLOY_LOGI(VX_ZONE_GRAPH, "############################\n");

  // NNDEPLOY_LOGI(VX_ZONE_GRAPH, "############################\n");
  // NNDEPLOY_LOGI(VX_ZONE_GRAPH, "Optimizer Graph V1!\n");
  // NNDEPLOY_LOGI(VX_ZONE_GRAPH, "############################\n");

  // NNDEPLOY_LOGI(VX_ZONE_GRAPH, "#########################\n");
  // NNDEPLOY_LOGI(VX_ZONE_GRAPH, "Device Verification Phase!\n");
  // NNDEPLOY_LOGI(VX_ZONE_GRAPH, "#########################\n");

  // NNDEPLOY_LOGI(VX_ZONE_GRAPH, "############################\n");
  // NNDEPLOY_LOGI(VX_ZONE_GRAPH, "Optimizer Graph V2!\n");
  // NNDEPLOY_LOGI(VX_ZONE_GRAPH, "############################\n");

  NNDEPLOY_LOGI(VX_ZONE_GRAPH, "#######################\n");
  NNDEPLOY_LOGI(VX_ZONE_GRAPH, "Task Initialize Phase!\n");
  NNDEPLOY_LOGI(VX_ZONE_GRAPH, "#######################\n");
  for (auto task_vec : topo_sort_task_) {
    for (auto task : task_vec) {
      status = task->init();
      if (status != base::kStatusCodeOK) {
        NNDEPLOY_LOG_ERROR("Task init failed!\n");
        return status;
      }
    }
  }

  // NNDEPLOY_LOGI(VX_ZONE_GRAPH, "########################\n");
  // NNDEPLOY_LOGI(VX_ZONE_GRAPH, "Memory Allocation Phase!\n");
  // NNDEPLOY_LOGI(VX_ZONE_GRAPH, "########################\n");

  // NNDEPLOY_LOGI(VX_ZONE_GRAPH, "#######################\n");
  // NNDEPLOY_LOGI(VX_ZONE_GRAPH, "Cost Calculations!\n");
  // NNDEPLOY_LOGI(VX_ZONE_GRAPH, "#######################\n");

  return status;
}

base::Status Pipeline::deinit() {
  NNDEPLOY_LOGI(VX_ZONE_GRAPH, "#######################\n");
  NNDEPLOY_LOGI(VX_ZONE_GRAPH, "Task DeInitialize Phase!\n");
  NNDEPLOY_LOGI(VX_ZONE_GRAPH, "#######################\n");
  for (auto task_vec : topo_sort_task_) {
    for (auto task : task_vec) {
      status = task->deinit();
      if (status != base::kStatusCodeOK) {
        NNDEPLOY_LOG_ERROR("Task deinit failed!\n");
        return status;
      }
    }
  }
}

base::ShapeMap Pipeline::inferOuputShape() {
  NNDEPLOY_LOGI(VX_ZONE_GRAPH, "#######################\n");
  NNDEPLOY_LOGI(VX_ZONE_GRAPH, "Task DeInitialize Phase!\n");
  NNDEPLOY_LOGI(VX_ZONE_GRAPH, "#######################\n");
  for (auto task_vec : topo_sort_task_) {
    for (auto task : task_vec) {
      status = task->inferOuputShape();
      if (status != base::kStatusCodeOK) {
        NNDEPLOY_LOG_ERROR("Task inferOuputShape failed!\n");
        return status;
      }
    }
  }
}

base::Status Pipeline::run() {
  NNDEPLOY_LOGI(VX_ZONE_GRAPH, "#######################\n");
  NNDEPLOY_LOGI(VX_ZONE_GRAPH, "Task DeInitialize Phase!\n");
  NNDEPLOY_LOGI(VX_ZONE_GRAPH, "#######################\n");
  for (auto task_vec : topo_sort_task_) {
    for (auto task : task_vec) {
      status = task->run();
      if (status != base::kStatusCodeOK) {
        NNDEPLOY_LOG_ERROR("Task run failed!\n");
        return status;
      }
    }
  }
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
task_wrapper* Pipeline::findTaskWrapper(Task* task) {
  for (auto task_wrapper : task_repository_) {
    if (task_wrapper->task_ == task) {
      return task_wrapper;
    }
  }
  return nullptr;
}

std::vector<task_wrapper*> Pipeline::findStartTasks() {
  std::vector<task_wrapper*> start_tasks;
  for (auto task_wrapper : task_repository_) {
    if (task_wrapper->predecessors_.empty()) {
      start_tasks.emplace_back(task_wrapper);
    }
  }
  return start_tasks;
}

std::vector<task_wrapper*> Pipeline::findEndTasks() {
  std::vector<task_wrapper*> end_tasks;
  for (auto task_wrapper : task_repository_) {
    if (task_wrapper->successors_.empty()) {
      end_tasks.emplace_back(task_wrapper);
    }
  }
  return end_tasks;
}

base::Status Pipeline::TopoSortBFS(task_wrapper* task_wrapper) {
  std::vector<Task*> dst;
  task_wrapper->color_ = kTaskColorGray;
  std::deque<task_wrapper*> task_deque;
  task_deque.emplace_back(task_wrapper);
  while (!task_deque.empty()) {
    task_wrapper* task_wrapper = task_deque.front();
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
        NNDEPLOY_LOG_ERROR("Cycle detected in pipeline");
        return base::kStatusCodeErrorInvalidValue;
      } else if (successor->color_ == kTaskColorWhite) {
        successor->color_ = kTaskColorGray;
        task_deque.emplace_back(successor);
      }
    }
    task_deque.pop_front();
    task_wrapper->color_ = 2;
    dst.emplace_back(task_wrapper->task_);
  }
  topo_sort_tasks_.pushu_back(dst);
  return base::kStatusCodeOK;
}

base::Status Pipeline::TopoSortDFS(task_wrapper* task_wrapper,
                                   std::stack<task_wrapper*> dst) {
  base::Status status = base::kStatusCodeOk;
  task_wrapper->color_ = kTaskColorGray;
  for (auto successor : task_wrapper->successors_) {
    if (successor->color_ == kTaskColorWhite) {
      status = TopoSortDFS(successor);
    } else if (successor->color_ == kTaskColorGray) {
      NNDEPLOY_LOG_ERROR("Cycle detected in pipeline");
      status = base::kStatusCodeErrorInvalidValue;
    } else {
      continue;
    }
  }
  if (status != base::kStatusCodeOK) {
    return status;
  }
  task_wrapper->color_ = kTaskColorBlack;
  dst.push(task_wrapper->task_);
  return base::kStatusCodeOK;
}

/**
 * @brief topo sort and check cycle
 *
 * @return base::Status
 */
base::Status Pipeline::TopologicalSort() {
  base::Status status = base::kStatusCodeOK;

  std::vector<task_wrapper*> start_tasks = findStartTasks();
  if (start_tasks.empty()) {
    NNDEPLOY_LOG_ERROR("No start task found in pipeline");
    return base::kStatusCodeErrorInvalidValue;
  }
  PipelineParam* param = dynamic_cast<PipelineParam*>(this->param_.get());
  if (param->topo_sort_type_ == PipelineParam::kTopoSortTypeBFS) {
    for (auto task_wrapper : start_tasks) {
      if (task_wrapper->color_ == kTaskColorBlack) {
        continue;
      }
      status = TopoSortBFS(task_wrapper);
      if (status != base::kStatusCodeOK) {
        NNDEPLOY_LOG_ERROR("TopoSortBFS failed");
        return status;
      }
    }
  } else {
    std::stack<task_wrapper*> dst;
    for (auto task_wrapper : start_tasks) {
      if (task_wrapper->color_ == kTaskColorBlack) {
        continue;
      }
      status = TopoSortDFS(task_wrapper, dst);
      if (status != base::kStatusCodeOK) {
        NNDEPLOY_LOG_ERROR("TopoSortDFS failed");
        return status;
      }
    }
    std::vector<Task*> task_dst;
    while (!dst.empty()) {
      task_dst.emplace_back(dst.top()->task_);
      dst.pop();
    }
    topo_sort_tasks_.push_back(task_dst);
  }

  return status;
}

}  // namespace task
}  // namespace nndeploy

#endif  // _NNDEPLOY_SOURCE_GRAPH_GRAPH_H_
