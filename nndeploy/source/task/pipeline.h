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

class PipelineParam : public base::Param {};

class Pipeline : public Task {
 public:
  Pipeline(const std::string& name, Packet* input, Packet* output);
  virtual ~Pipeline();

  Packet* createPacket(const std::string& name = "", Packet* input = nullptr,
                       Packet* output = nullptr);
  void addPacket(Packet* edge);

  template <typename T>
  Task* createTask(const std::string& name, Packet* input, Packet* output);
  /**
   * @brief
   *
   * @param task 必须确保的输入输出Packet已经设置好
   * @return base::Status
   */
  virtual base::Status addTask(Task* task);

  // virtual base::Status setName(const std::string& name);
  // virtual std::string getName();

  // virtual base::Status setParam(base::Param* param);
  // virtual base::Param* getParam();

  // virtual Packet* getInput();
  // virtual Packet* getOutput();

  // virtual base::Status setInput(Packet* input);
  // virtual base::Status setOutput(Packet* output);

  virtual base::Status init();
  virtual base::Status deinit();

  virtual base::ShapeMap inferOuputShape();

  virtual base::Status run();

  // virtual base::Status dump(std::ostream& oss = std::cout);

 protected:
  std::vector<PacketWrapper*> packet_repository_;
  std::vector<TaskWrapper*> task_repository_;

  std::vector<Task*> exe_task_list_;
};

}  // namespace task
}  // namespace nndeploy

#endif  // _NNDEPLOY_SOURCE_TASK_PIPELINE_H_
