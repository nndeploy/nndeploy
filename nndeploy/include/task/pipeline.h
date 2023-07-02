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
#include "nndeploy/source/task/packet.h"
#include "nndeploy/source/task/task.h"

namespace nndeploy {
namespace task {

class PacketWrapper;

class TaskWrapper;

enum TopoSortType : int32_t { kTopoSortTypeBFS = 0x0000, kTopoSortTypeDFS };

class PipelineParam : public base::Param {
 public:
  TopoSortType topo_sort_type_ = kTopoSortTypeDFS;
};

class Pipeline : public Task {
 public:
  Pipeline(const std::string& name, Packet* input, Packet* output);
  Pipeline(const std::string& name, std::vector<Packet*> inputs,
           std::vector<Packet*> outputs);
  virtual ~Pipeline();

  virtual Packet* createPacket(const std::string& name = "");
  virtual base::Status addPacket(Packet* packet);

  template <typename T>
  virtual Task* createTask(const std::string& name, Packet* input,
                           Packet* output);
  virtual Task* createTask(const std::string& name, std::vector<Packet*> inputs,
                           std::vector<Packet*> outputs);
  virtual base::Status addTask(Task* task);

  virtual base::Status init();
  virtual base::Status deinit();

  virtual base::ShapeMap inferOuputShape();

  virtual base::Status run();

  // virtual base::Status dump(std::ostream& oss = std::cout);

 protected:
  PacketWrapper* findPacketWrapper(Packet* packet);
  task_wrapper* findTaskWrapper(Task* task);

  std::vector<task_wrapper*> findStartTasks();
  std::vector<task_wrapper*> findEndTasks();

  base::Status TopoSortBFS(task_wrapper* task_wrapper);
  base::Status TopoSortDFS(task_wrapper* task_wrapper,
                           std::stack<task_wrapper*> dst);
  base::Status topologicalSort();

 protected:
  std::vector<PacketWrapper*> packet_repository_;
  std::vector<TaskWrapper*> task_repository_;

  std::vector<std::vector<Task*>> topo_sort_task_;
};

}  // namespace task
}  // namespace nndeploy

#endif  // _NNDEPLOY_SOURCE_TASK_PIPELINE_H_
