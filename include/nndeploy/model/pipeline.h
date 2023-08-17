#ifndef _NNDEPLOY_MODEL_PIPELINE_H_
#define _NNDEPLOY_MODEL_PIPELINE_H_

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
#include "nndeploy/model/infer.h"
#include "nndeploy/model/packet.h"
#include "nndeploy/model/task.h"

namespace nndeploy {
namespace model {

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
  PacketWrapper* addPacket(Packet* packet);

  template <typename T,
            typename std::enable_if<std::is_base_of<Task, T>{}, int>::type = 0>
  Task* createTask(const std::string& name, Packet* input, Packet* output) {
    NNDEPLOY_CHECK_PARAM_NULL_RET_NULL(input, "input is null!");
    NNDEPLOY_CHECK_PARAM_NULL_RET_NULL(output, "output is null!");
    Task* task = dynamic_cast<Task*>(new T(name, input, output));
    TaskWrapper* task_wrapper = new TaskWrapper();
    task_wrapper->is_external_ = false;
    task_wrapper->task_ = task;
    task_wrapper->name_ = name;
    PacketWrapper* input_wrapper = findPacketWrapper(input);
    if (findPacketWrapper(input) == nullptr) {
      input_wrapper = this->addPacket(input);
    }
    input_wrapper->consumers_.emplace_back(task_wrapper);
    PacketWrapper* output_wrapper = findPacketWrapper(output);
    if (output_wrapper == nullptr) {
      output_wrapper = this->addPacket(output);
    }
    output_wrapper->producers_.emplace_back(task_wrapper);

    task_repository_.emplace_back(task_wrapper);
    return task;
  }
  template <typename T,
            typename std::enable_if<std::is_base_of<Task, T>{}, int>::type = 0>
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
      PacketWrapper* input_wrapper = findPacketWrapper(input);
      if (findPacketWrapper(input) == nullptr) {
        input_wrapper = this->addPacket(input);
      }
      input_wrapper->consumers_.emplace_back(task_wrapper);
    }
    for (auto output : outputs) {
      PacketWrapper* output_wrapper = findPacketWrapper(output);
      if (output_wrapper == nullptr) {
        output_wrapper = this->addPacket(output);
      }
      output_wrapper->producers_.emplace_back(task_wrapper);
    }

    task_repository_.emplace_back(task_wrapper);
    return task;
  }
  template <typename T,
            typename std::enable_if<std::is_base_of<Task, T>{}, int>::type = 0>
  Task* createInfer(const std::string& name, base::InferenceType type,
                    Packet* input, Packet* output) {
    NNDEPLOY_CHECK_PARAM_NULL_RET_NULL(input, "input is null!");
    NNDEPLOY_CHECK_PARAM_NULL_RET_NULL(output, "output is null!");
    Task* task = dynamic_cast<Task*>(new T(name, type, input, output));
    TaskWrapper* task_wrapper = new TaskWrapper();
    task_wrapper->is_external_ = false;
    task_wrapper->task_ = task;
    task_wrapper->name_ = name;
    PacketWrapper* input_wrapper = findPacketWrapper(input);
    if (findPacketWrapper(input) == nullptr) {
      input_wrapper = this->addPacket(input);
    }
    input_wrapper->consumers_.emplace_back(task_wrapper);
    PacketWrapper* output_wrapper = findPacketWrapper(output);
    if (output_wrapper == nullptr) {
      output_wrapper = this->addPacket(output);
    }
    output_wrapper->producers_.emplace_back(task_wrapper);

    task_repository_.emplace_back(task_wrapper);
    return task;
  }
  base::Status addTask(Task* task);

  base::Status setTaskParam(const std::string& task_name, base::Param* param);
  base::Param* getTaskParam(const std::string& task_name);

  base::Status init();
  base::Status deinit();

  base::Status reshape();

  base::Status run();

  base::Status dump(std::ostream& oss = std::cout);

 protected:
  PacketWrapper* findPacketWrapper(Packet* packet);
  TaskWrapper* findTaskWrapper(const std::string& task_name);
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

using creatPipelineFunc = std::function<Pipeline*(
    const std::string& name, base::InferenceType inference_type,
    base::DeviceType device_type, Packet* input, Packet* output,
    base::ModelType model_type, bool is_path,
    std::vector<std::string>& model_value, base::EncryptType encrypt_type)>;

std::map<std::string, creatPipelineFunc>& getGlobalPipelineCreatorMap();

class TypePipelineRegister {
 public:
  explicit TypePipelineRegister(const std::string& name,
                                creatPipelineFunc func) {
    getGlobalPipelineCreatorMap()[name] = func;
  }
};

Pipeline* creatPipeline(
    const std::string& name, base::InferenceType inference_type,
    base::DeviceType device_type, Packet* input, Packet* output,
    base::ModelType model_type, bool is_path,
    std::vector<std::string>& model_value,
    base::EncryptType encrypt_type = base::kEncryptTypeNone);

}  // namespace model
}  // namespace nndeploy

#endif  // _NNDEPLOY_MODEL_PIPELINE_H_
