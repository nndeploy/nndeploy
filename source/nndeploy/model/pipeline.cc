
#include "nndeploy/model/pipeline.h"

#include "nndeploy/base/common.h"
#include "nndeploy/base/glic_stl_include.h"
#include "nndeploy/base/log.h"
#include "nndeploy/base/macro.h"
#include "nndeploy/base/object.h"
#include "nndeploy/base/status.h"
#include "nndeploy/base/string.h"
#include "nndeploy/base/time_profiler.h"
#include "nndeploy/base/value.h"
#include "nndeploy/device/buffer.h"
#include "nndeploy/device/buffer_pool.h"
#include "nndeploy/device/device.h"
#include "nndeploy/device/tensor.h"
#include "nndeploy/inference/inference.h"
#include "nndeploy/inference/inference_param.h"
#include "nndeploy/model/packet.h"
#include "nndeploy/model/task.h"

namespace nndeploy {
namespace model {

Pipeline::Pipeline(const std::string& name, Packet* input, Packet* output)
    : Task(name, input, output) {
  param_ = std::make_shared<PipelineParam>();
  if (nullptr == addPacket(input)) {
    constructed_ = false;
    return;
  }
  if (nullptr == addPacket(output)) {
    constructed_ = false;
    return;
  }
  constructed_ = true;
}
Pipeline::Pipeline(const std::string& name, std::vector<Packet*> inputs,
                   std::vector<Packet*> outputs)
    : Task(name, inputs, outputs) {
  param_ = std::make_shared<PipelineParam>();
  for (auto input : inputs) {
    if (nullptr == addPacket(input)) {
      constructed_ = false;
      return;
    }
  }
  for (auto output : outputs) {
    if (nullptr == addPacket(output)) {
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
PacketWrapper* Pipeline::addPacket(Packet* packet) {
  base::Status status = base::kStatusCodeOk;
  NNDEPLOY_CHECK_PARAM_NULL_RET_NULL(packet, "packet is null!");
  PacketWrapper* packet_wrapper = new PacketWrapper();
  packet_wrapper->is_external_ = true;
  packet_wrapper->packet_ = packet;
  packet_repository_.emplace_back(packet_wrapper);
  return packet_wrapper;
}

// template <typename T>
// Task* Pipeline::createTask(const std::string& name, Packet* input,
//                            Packet* output) {
//   NNDEPLOY_CHECK_PARAM_NULL_RET_NULL(input, "input is null!");
//   NNDEPLOY_CHECK_PARAM_NULL_RET_NULL(output, "output is null!");
//   Task* task = dynamic_cast<Task*>(new T(name, input, output));
//   TaskWrapper* task_wrapper = new TaskWrapper();
//   task_wrapper->is_external_ = false;
//   task_wrapper->task_ = task;
//   task_wrapper->name_ = name;
//   if (findPacketWrapper(input) == nullptr) {
//     this->addPacket(input);
//   }
//   findPacketWrapper(input)->consumers_.emplace_back(task_wrapper);
//   if (findPacketWrapper(output) == nullptr) {
//     this->addPacket(output);
//   }
//   findPacketWrapper(output)->producers_.emplace_back(task_wrapper);
//   task_repository_.emplace_back(task_wrapper);
//   return task;
// }
// template <typename T>
// Task* Pipeline::createTask(const std::string& name, std::vector<Packet*>
// inputs,
//                            std::vector<Packet*> outputs) {
//   if (inputs.empty() || outputs.empty()) {
//     NNDEPLOY_LOGE("inputs or outputs is empty!\n");
//     return nullptr;
//   }
//   Task* task = dynamic_cast<Task*>(new T(name, inputs, outputs));
//   TaskWrapper* task_wrapper = new TaskWrapper();
//   task_wrapper->is_external_ = false;
//   task_wrapper->task_ = task;
//   task_wrapper->name_ = name;
//   for (auto input : inputs) {
//     if (findPacketWrapper(input) == nullptr) {
//       this->addPacket(input);
//     }
//     findPacketWrapper(input)->consumers_.emplace_back(task_wrapper);
//   }
//   for (auto output : outputs) {
//     if (findPacketWrapper(output) == nullptr) {
//       this->addPacket(output);
//     }
//     findPacketWrapper(output)->producers_.emplace_back(task_wrapper);
//   }
//   task_repository_.emplace_back(task_wrapper);
//   return task;
// }
// template <typename T>
// Task* Pipeline::createInfer(const std::string& name, base::InferenceType
// type,
//                             Packet* input, Packet* output) {
//   NNDEPLOY_CHECK_PARAM_NULL_RET_NULL(input, "input is null!");
//   NNDEPLOY_CHECK_PARAM_NULL_RET_NULL(output, "output is null!");
//   Task* task = dynamic_cast<Task*>(new T(name, type, input, output));
//   TaskWrapper* task_wrapper = new TaskWrapper();
//   task_wrapper->is_external_ = false;
//   task_wrapper->task_ = task;
//   task_wrapper->name_ = name;
//   if (findPacketWrapper(input) == nullptr) {
//     this->addPacket(input);
//   }
//   findPacketWrapper(input)->consumers_.emplace_back(task_wrapper);
//   if (findPacketWrapper(output) == nullptr) {
//     this->addPacket(output);
//   }
//   findPacketWrapper(output)->producers_.emplace_back(task_wrapper);
//   task_repository_.emplace_back(task_wrapper);
//   return task;
// }
base::Status Pipeline::addTask(Task* task) {
  base::Status status = base::kStatusCodeOk;
  NNDEPLOY_CHECK_PARAM_NULL_RET_STATUS(task, "task is null!");
  TaskWrapper* task_wrapper = new TaskWrapper();
  task_wrapper->is_external_ = true;
  task_wrapper->task_ = task;
  task_wrapper->name_ = task->getName();
  for (auto input : task->getAllInput()) {
    PacketWrapper* input_wrapper = findPacketWrapper(input);
    if (findPacketWrapper(input) == nullptr) {
      input_wrapper = this->addPacket(input);
    }
    input_wrapper->consumers_.emplace_back(task_wrapper);
  }
  for (auto output : task->getAllOutput()) {
    PacketWrapper* output_wrapper = findPacketWrapper(output);
    if (output_wrapper == nullptr) {
      output_wrapper = this->addPacket(output);
    }
    output_wrapper->producers_.emplace_back(task_wrapper);
  }

  task_repository_.emplace_back(task_wrapper);
  return status;
}

base::Status Pipeline::setTaskParam(const std::string& task_name,
                                    base::Param* param) {
  base::Status status = base::kStatusCodeOk;
  NNDEPLOY_CHECK_PARAM_NULL_RET_STATUS(param, "param is null!");
  TaskWrapper* task_wrapper = findTaskWrapper(task_name);
  NNDEPLOY_CHECK_PARAM_NULL_RET_STATUS(task_wrapper, "task_wrapper is null!");
  status = task_wrapper->task_->setParam(param);
  return status;
}

base::Param* Pipeline::getTaskParam(const std::string& task_name) {
  TaskWrapper* task_wrapper = findTaskWrapper(task_name);
  NNDEPLOY_CHECK_PARAM_NULL_RET_NULL(task_wrapper, "task_wrapper is null!");
  return task_wrapper->task_->getParam();
}

base::Status Pipeline::init() {
  base::Status status = base::kStatusCodeOk;

  // NNDEPLOY_LOGI("###########################\n");
  // NNDEPLOY_LOGI("Parameter Validation Phase!\n");
  // NNDEPLOY_LOGI("###########################\n");
  for (auto task_wrapper : task_repository_) {
    NNDEPLOY_CHECK_PARAM_NULL_RET_STATUS(task_wrapper->task_,
                                         "packet_repository_ task is null!");
  }
  for (auto packet_wrapper : packet_repository_) {
    NNDEPLOY_CHECK_PARAM_NULL_RET_STATUS(packet_wrapper->packet_,
                                         "packet_repository_ packet is null!");
  }

  // NNDEPLOY_LOGI("####################\n");
  // NNDEPLOY_LOGI("Mark Predecessors And Successors Phase!\n");
  // NNDEPLOY_LOGI("####################\n");
  for (auto task_wrapper : task_repository_) {
    Task* task = task_wrapper->task_;
    std::vector<Packet*> inputs = task->getAllInput();
    for (auto input : inputs) {
      PacketWrapper* input_wrapper = findPacketWrapper(input);
      NNDEPLOY_CHECK_PARAM_NULL_RET_STATUS(input_wrapper,
                                           "input_wrapper is null!");
      task_wrapper->predecessors_.assign(input_wrapper->producers_.begin(),
                                         input_wrapper->producers_.end());
    }
    std::vector<Packet*> outputs = task->getAllOutput();
    for (auto output : outputs) {
      PacketWrapper* output_wrapper = findPacketWrapper(output);
      NNDEPLOY_CHECK_PARAM_NULL_RET_STATUS(output_wrapper,
                                           "output_wrapper is null!");
      task_wrapper->successors_.assign(output_wrapper->consumers_.begin(),
                                       output_wrapper->consumers_.end());
    }
  }

  // NNDEPLOY_LOGI("##############\n");
  // NNDEPLOY_LOGI("TopologicalSort and Check Cycle!\n");
  // NNDEPLOY_LOGI("##############\n");
  status = topologicalSort();
  if (status != base::kStatusCodeOk) {
    NNDEPLOY_LOGE("Toposort failed");
    return status;
  }

  // // NNDEPLOY_LOGI("############################\n");
  // // NNDEPLOY_LOGI("Checking for Unvisited Packet!\n");
  // // NNDEPLOY_LOGI("############################\n");

  // // NNDEPLOY_LOGI("############################\n");
  // // NNDEPLOY_LOGI("Optimizer Graph V1!\n");
  // // NNDEPLOY_LOGI("############################\n");

  // // NNDEPLOY_LOGI("#########################\n");
  // // NNDEPLOY_LOGI("Device Verification Phase!\n");
  // // NNDEPLOY_LOGI("#########################\n");

  // // NNDEPLOY_LOGI("############################\n");
  // // NNDEPLOY_LOGI("Optimizer Graph V2!\n");
  // // NNDEPLOY_LOGI("############################\n");

  // NNDEPLOY_LOGI("#######################\n");
  // NNDEPLOY_LOGI("Task Initialize Phase!\n");
  // NNDEPLOY_LOGI("#######################\n");
  for (auto task_vec : topo_sort_task_) {
    for (auto task : task_vec) {
      status = task->init();
      if (status != base::kStatusCodeOk) {
        NNDEPLOY_LOGE("Task init failed!\n");
        return status;
      }
    }
  }

  // // NNDEPLOY_LOGI("########################\n");
  // // NNDEPLOY_LOGI("Memory Allocation Phase!\n");
  // // NNDEPLOY_LOGI("########################\n");

  // // NNDEPLOY_LOGI("#######################\n");
  // // NNDEPLOY_LOGI("Cost Calculations!\n");
  // // NNDEPLOY_LOGI("#######################\n");

  return status;
}

base::Status Pipeline::deinit() {
  base::Status status = base::kStatusCodeOk;
  // NNDEPLOY_LOGI("#######################\n");
  // NNDEPLOY_LOGI("Task DeInitialize Phase!\n");
  // NNDEPLOY_LOGI("#######################\n");
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

base::Status Pipeline::reshape() {
  base::Status status = base::kStatusCodeOk;
  // NNDEPLOY_LOGI("#######################\n");
  // NNDEPLOY_LOGI("Task reshape Phase!\n");
  // NNDEPLOY_LOGI("#######################\n");
  for (auto task_vec : topo_sort_task_) {
    for (auto task : task_vec) {
      status = task->reshape();
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
  is_running_ = true;
  // NNDEPLOY_LOGI("#######################\n");
  // NNDEPLOY_LOGI("Task run Phase!\n");
  // NNDEPLOY_LOGI("#######################\n");
  for (auto task_vec : topo_sort_task_) {
    for (auto task : task_vec) {
      // NNDEPLOY_LOGI("Task name[%s]!\n", task->getName().c_str());
      NNDEPLOY_TIME_POINT_START(task->getName());
      status = task->run();
      NNDEPLOY_TIME_POINT_END(task->getName());
      if (status != base::kStatusCodeOk) {
        NNDEPLOY_LOGE("Task run failed!\n");
        return status;
      }
    }
  }
  return status;
}

base::Status Pipeline::dump(std::ostream& oss) {
  base::Status status = base::kStatusCodeOk;
  // NNDEPLOY_LOGI("#######################\n");
  // NNDEPLOY_LOGI("Task dump Phase!\n");
  // NNDEPLOY_LOGI("#######################\n");
  if (name_.empty()) {
    oss << "digraph pipeline {\n";
  } else {
    oss << "digraph " << name_ << " {\n";
  }
  for (auto task_vec : topo_sort_task_) {
    for (auto task : task_vec) {
      TaskWrapper* task_wrapper = findTaskWrapper(task);
      if (task_wrapper->predecessors_.empty()) {
        auto inputs = task->getAllInput();
        for (auto input : inputs) {
          oss << "p" << (void*)input << "[label=input]\n";
          oss << "p" << (void*)input << "->"
              << "p" << (void*)task;
          if (input->getName().empty()) {
            oss << "\n";
          } else {
            oss << "[label=" << input->getName() << "]\n";
          }
        }
      }
      if (task->getName().empty()) {
        oss << "p" << (void*)task << "\n";
      } else {
        oss << "p" << (void*)task << "[label=" << task->getName() << "]\n";
      }
      if (task_wrapper->successors_.empty()) {
        auto outputs = task->getAllOutput();
        for (auto output : outputs) {
          oss << "p" << (void*)output << "[label=output]\n";
          oss << "p" << (void*)task << "->"
              << "p" << (void*)output;
          if (output->getName().empty()) {
            oss << "\n";
          } else {
            oss << "[label=" << output->getName() << "]\n";
          }
        }
      } else {
        for (auto successor : task_wrapper->successors_) {
          oss << "p" << (void*)task << "->"
              << "p" << (void*)(successor->task_);
          auto outputs = task->getAllOutput();
          auto inputs = successor->task_->getAllInput();
          Packet* out_in = nullptr;
          for (auto output : outputs) {
            for (auto input : inputs) {
              if (output == input) {
                out_in = output;
              }
            }
          }
          if (out_in != nullptr) {
            if (out_in->getName().empty()) {
              oss << "\n";
            } else {
              oss << "[label=" << out_in->getName() << "]\n";
            }
          }
        }
      }
    }
  }
  oss << "}\n";
  return status;
}

PacketWrapper* Pipeline::findPacketWrapper(Packet* packet) {
  for (auto packet_wrapper : packet_repository_) {
    if (packet_wrapper->packet_ == packet) {
      return packet_wrapper;
    }
  }
  return nullptr;
}
TaskWrapper* Pipeline::findTaskWrapper(const std::string& task_name) {
  for (auto task_wrapper : task_repository_) {
    if (task_wrapper->name_ == task_name) {
      return task_wrapper;
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

std::map<std::string, createPipelineFunc>& getGlobalPipelineCreatorMap() {
  static std::once_flag once;
  static std::shared_ptr<std::map<std::string, createPipelineFunc>> creators;
  std::call_once(once, []() {
    creators.reset(new std::map<std::string, createPipelineFunc>);
  });
  return *creators;
}

Pipeline* createPipeline(const std::string& name,
                         base::InferenceType inference_type,
                         base::DeviceType device_type, Packet* input,
                         Packet* output, base::ModelType model_type,
                         bool is_path, std::vector<std::string>& model_value) {
  Pipeline* temp = nullptr;
  auto& creater_map = getGlobalPipelineCreatorMap();
  if (creater_map.count(name) > 0) {
    temp = creater_map[name](name, inference_type, device_type, input, output,
                             model_type, is_path, model_value);
  }
  return temp;
}

}  // namespace model
}  // namespace nndeploy
