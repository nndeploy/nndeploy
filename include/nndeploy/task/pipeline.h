#ifndef _NNDEPLOY_TASK_PIPELINE_H_
#define _NNDEPLOY_TASK_PIPELINE_H_

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
#include "nndeploy/inference/inference.h"
#include "nndeploy/inference/inference_param.h"
#include "nndeploy/task/packet.h"
#include "nndeploy/task/task.h"

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

#endif  // _NNDEPLOY_TASK_PIPELINE_H_
