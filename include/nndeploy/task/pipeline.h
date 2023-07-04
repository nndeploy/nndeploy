#ifndef _NNDEPLOY_TASK_PIPELINE_H_
#define _NNDEPLOY_TASK_PIPELINE_H_

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
#include "nndeploy/task/inference/inference.h"
#include "nndeploy/task/packet.h"
#include "nndeploy/task/task.h"

namespace nndeploy {
namespace task {

class TaskWrapper;

class PacketWrapper;

enum TopoSortType : int { kTopoSortTypeBFS = 0x0000, kTopoSortTypeDFS };

class PipelineParam : public base::Param {
 public:
  TopoSortType topo_sort_type_ = kTopoSortTypeDFS;
};

class Pipeline : public Task {
 public:
  Pipeline(const std::string& name, Packet* input, Packet* output);
  Pipeline(const std::string& name, std::vector<Packet*> inputs,
           std::vector<Packet*> outputs);
  ~Pipeline();

  Packet* createPacket(const std::string& name = "");
  base::Status addPacket(Packet* packet);

  template <typename T>
  Task* createTask(const std::string& name, Packet* input, Packet* output);
  template <typename T>
  Task* createTask(const std::string& name, std::vector<Packet*> inputs,
                   std::vector<Packet*> outputs);
  template <typename T>
  Task* createInference(const std::string& name, base::InferenceType type,
                        Packet* input, Packet* output);
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

}  // namespace task
}  // namespace nndeploy

#endif  // _NNDEPLOY_TASK_PIPELINE_H_
