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

class PacketWrapper {
 public:
  bool is_external_;
  Packet* packet_;
  std::vector<Task*> producers_;
  std::vector<Task*> consumers_;
}

class TaskWrapper {
 public:
  bool is_external_;
  Task* task_;
  std::string name_;
  Packet* input_ = nullptr;
  Packet* output_ = nullptr;
  std::vector<Task*> predecessors_;
  std::vector<Task*> successors_;
}

Pipeline::Pipeline(const std::string& name, Packet* input, Packet* output)
    : Task(name, input, output) {
  param_ = std::make_shared<PipelineParam>();
  addPacket(input);
  addPacket(output);
}
~Pipeline::Pipeline() {}

Packet* Pipeline::createPacket(const std::string& name) {
  Packet* packet = new Packet(name);
  PacketWrapper* packet_wrapper = new PacketWrapper();
  packet_wrapper->is_external_ = false;
  packet_wrapper->packet_ = packet;
  packet_repository_.emplace_back(packet_wrapper);
  return packet;
}
void Pipeline::addPacket(Packet* packet) {
  PacketWrapper* packet_wrapper = new PacketWrapper();
  packet_wrapper->is_external_ = true;
  packet_wrapper->packet_ = packet;
  packet_repository_.emplace_back(packet_wrapper);
}

template <typename T>
Task* Pipeline::createTask(const std::string& name, Packet* input,
                           Packet* output) {
  Task* task = new T(name, param, input, output);
  TaskWrapper* task_wrapper = new TaskWrapper();
  task_wrapper->is_external_ = false;
  task_wrapper->task_ = task;
  task_wrapper->name_ = name;
  task_wrapper->input_ = input;
  if (findPacketWrapper(input) == nullptr) {
    this->addPacket(input);
  }
  findPacketWrapper(input)->consumers_.emplace_back(task);
  task_wrapper->output_ = output;
  if (findPacketWrapper(output) == nullptr) {
    this->addPacket(output);
  }
  findPacketWrapper(output)->producers_.emplace_back(task);
  task_repository_.emplace_back(task_wrapper);
  return dynamic_cast<Task*> task;
}
void Pipeline::addTask(Task* task) {
  TaskWrapper* task_wrapper = new TaskWrapper();
  task_wrapper->is_external_ = true;
  task_wrapper->task_ = task;
  task_wrapper->name_ = task->getName();
  Packet* input = task->getInput();
  task_wrapper->input_ = input;
  if (findPacketWrapper(input) == nullptr) {
    this->addPacket(input);
  }
  findPacketWrapper(input)->consumers_.emplace_back(task);
  Packet* output = task->getOutput();
  if (findPacketWrapper(output) == nullptr) {
    this->addPacket(output);
  }
  findPacketWrapper(output)->producers_.emplace_back(task);
  task_wrapper->output_ = output;
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
    PacketWrapper* input = findPacketWrapper(task_wrapper->input_);
    task_wrapper->predecessors_.assign(input->producers_.begin(),
                                       input->producers_.end());
    PacketWrapper* output = findPacketWrapper(task_wrapper->output_);
    task_wrapper->successors_.assign(output->consumers_.begin(),
                                     output->consumers_.end());
  }

  NNDEPLOY_LOGI(VX_ZONE_GRAPH, "###############################\n");
  NNDEPLOY_LOGI(VX_ZONE_GRAPH, "Find Start Task Phase!\n");
  NNDEPLOY_LOGI(VX_ZONE_GRAPH, "###############################\n");
  // find start
  std::vector<task_wrapper*> start_tasks = findStartTasks();
  if (start_tasks.empty()) {
    NNDEPLOY_LOG_ERROR("No start task found in pipeline");
    return base::kStatusCodeErrorInvalidValue;
  }

  NNDEPLOY_LOGI(VX_ZONE_GRAPH, "##############\n");
  NNDEPLOY_LOGI(VX_ZONE_GRAPH, "Cycle Checking!\n");
  NNDEPLOY_LOGI(VX_ZONE_GRAPH, "##############\n");

  NNDEPLOY_LOGI(VX_ZONE_GRAPH, "############################\n");
  NNDEPLOY_LOGI(VX_ZONE_GRAPH, "Checking for Unvisited Task and Packet!\n");
  NNDEPLOY_LOGI(VX_ZONE_GRAPH, "############################\n");

  NNDEPLOY_LOGI(VX_ZONE_GRAPH, "############################\n");
  NNDEPLOY_LOGI(VX_ZONE_GRAPH, "Optimizer Graph!\n");
  NNDEPLOY_LOGI(VX_ZONE_GRAPH, "############################\n");

  NNDEPLOY_LOGI(VX_ZONE_GRAPH, "#########################\n");
  NNDEPLOY_LOGI(VX_ZONE_GRAPH, "Device Verification Phase!\n");
  NNDEPLOY_LOGI(VX_ZONE_GRAPH, "#########################\n");

  NNDEPLOY_LOGI(VX_ZONE_GRAPH, "############################\n");
  NNDEPLOY_LOGI(VX_ZONE_GRAPH, "Optimizer Graph!\n");
  NNDEPLOY_LOGI(VX_ZONE_GRAPH, "############################\n");

  NNDEPLOY_LOGI(VX_ZONE_GRAPH, "#######################\n");
  NNDEPLOY_LOGI(VX_ZONE_GRAPH, "Task Initialize Phase!\n");
  NNDEPLOY_LOGI(VX_ZONE_GRAPH, "#######################\n");

  NNDEPLOY_LOGI(VX_ZONE_GRAPH, "########################\n");
  NNDEPLOY_LOGI(VX_ZONE_GRAPH, "Memory Allocation Phase!\n");
  NNDEPLOY_LOGI(VX_ZONE_GRAPH, "########################\n");

  NNDEPLOY_LOGI(VX_ZONE_GRAPH, "#######################\n");
  NNDEPLOY_LOGI(VX_ZONE_GRAPH, "Cost Calculations!\n");
  NNDEPLOY_LOGI(VX_ZONE_GRAPH, "#######################\n");

  return status;
}

base::Status Pipeline::deinit() {}

base::ShapeMap Pipeline::inferOuputShape();

base::Status Pipeline::run();

// base::Status Pipeline::dump(std::ostream& oss = std::cout);

std::vector<task_wrapper*> Pipeline::findStartTasks() {
  std::vector<task_wrapper*> start_tasks;
  for (auto task_wrapper : task_repository_) {
    if (task_wrapper->predecessors_.empty()) {
      start_tasks.emplace_back(task_wrapper);
    }
  }
  return start_tasks;
}

base::Status Pipeline::isCyclicDFS() {
  std::vector<task_wrapper*> start_tasks = findStartTasks();
  for (auto start_task : start_tasks) {
    start_task->setVisited();
    if (isCyclicDFS(start_task, visited, visiting)) {
      return base::kStatusCodeErrorInvalidValue;
    }
  }
  return base::kStatusCodeOK;
}

base::Status Pipeline::traverse(task_wrapper* start) {
  std::vector<task_wrapper*> start_tasks;
  for (auto task_wrapper : task_repository_) {
    if (task_wrapper->predecessors_.empty()) {
      start_tasks.emplace_back(task_wrapper);
    }
  }
  return start_tasks;
}

}  // namespace task
}  // namespace nndeploy

#endif  // _NNDEPLOY_SOURCE_GRAPH_GRAPH_H_
