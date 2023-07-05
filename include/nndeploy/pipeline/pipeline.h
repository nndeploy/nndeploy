#ifndef _NNDEPLOY_PIPELINE_PIPELINE_H_
#define _NNDEPLOY_PIPELINE_PIPELINE_H_

#include "nndeploy/base/common.h"
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
#include "nndeploy/pipeline/infer.h"
#include "nndeploy/pipeline/packet.h"
#include "nndeploy/pipeline/task.h"

namespace nndeploy {
namespace pipeline {

enum TaskColorType : int {
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

enum TopoSortType : int { kTopoSortTypeBFS = 0x0000, kTopoSortTypeDFS };

class NNDEPLOY_CC_API PipelineParam : public base::Param {
 public:
  TopoSortType topo_sort_type_ = kTopoSortTypeDFS;
};

class NNDEPLOY_CC_API Pipeline : public Task {
 public:
  Pipeline(const std::string& name, Packet* input, Packet* output);
  Pipeline(const std::string& name, std::vector<Packet*> inputs,
           std::vector<Packet*> outputs);
  ~Pipeline();

  Packet* createPacket(const std::string& name = "");
  base::Status addPacket(Packet* packet);

  template <typename T>
  Task* createTask(const std::string& name, Packet* input, Packet* output) {
    NNDEPLOY_CHECK_PARAM_NULL_RET_NULL(input, "input is null!");
    NNDEPLOY_CHECK_PARAM_NULL_RET_NULL(output, "output is null!");
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
  Task* createTask(const std::string& name, std::vector<Packet*> inputs,
                   std::vector<Packet*> outputs) {
    if (inputs.empty() || outputs.empty()) {
      NNDEPLOY_LOGE("inputs or outputs is empty!\n");
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
  Task* createInfer(const std::string& name, base::InferenceType type,
                    Packet* input, Packet* output) {
    NNDEPLOY_CHECK_PARAM_NULL_RET_NULL(input, "input is null!");
    NNDEPLOY_CHECK_PARAM_NULL_RET_NULL(output, "output is null!");
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
  base::Status addTask(Task* task);

  base::Status init();
  base::Status deinit();

  base::Status reShape();

  base::Status run();

  //  base::Status dump(std::ostream& oss = std::cout);

 protected:
  PacketWrapper* findPacketWrapper(Packet* packet);
  TaskWrapper* findTaskWrapper(Task* task);

  std::vector<TaskWrapper*> findStartTasks();
  std::vector<TaskWrapper*> findEndTasks();

  base::Status TopoSortBFS(TaskWrapper* task_wrapper);
  base::Status TopoSortDFS(TaskWrapper* task_wrapper,
                           std::stack<TaskWrapper*>& dst);
  base::Status topologicalSort();

 protected:
  std::vector<PacketWrapper*> packet_repository_;
  std::vector<TaskWrapper*> task_repository_;

  std::vector<std::vector<Task*>> topo_sort_task_;
};

// using cretePipelineFunc = Pipeline* (*)(const std::string& name,
//                                         base::InferenceType type, Packet*
//                                         input, Packet* output);
//
// std::map<std::string, cretePipelineFunc>& getGlobalPipelineCreatorMap();
//
// class TypePipelineRegister {
//  public:
//   explicit TypePipelineRegister(const std::string& name, cretePipelineFuncs
//   func) {
//     getGlobalPipelineCreatorMap()[name] = func;
//   }
// };
//
// Pipeline* cretePipeline(const std::string& name, base::InferenceType type,
//                         Packet* input, Packet* output);

}  // namespace pipeline
}  // namespace nndeploy

#endif  // _NNDEPLOY_PIPELINE_PIPELINE_H_
