
#ifndef _NNDEPLOY_DAG_NODE_H_
#define _NNDEPLOY_DAG_NODE_H_

#include "nndeploy/base/common.h"
#include "nndeploy/base/glic_stl_include.h"
#include "nndeploy/base/log.h"
#include "nndeploy/base/macro.h"
#include "nndeploy/base/object.h"
#include "nndeploy/base/status.h"
#include "nndeploy/base/string.h"
#include "nndeploy/base/time_profiler.h"
#include "nndeploy/base/any.h"
#include "nndeploy/dag/edge.h"
#include "nndeploy/device/buffer.h"
#include "nndeploy/device/device.h"
#include "nndeploy/device/memory_pool.h"
#include "nndeploy/device/tensor.h"

namespace nndeploy {
namespace dag {

/**
 * @brief
 * @note Each node is responsible for allocating memory for it's output edges.
 */
class NNDEPLOY_CC_API Node {
 public:
  Node(const std::string &name, Edge *input, Edge *output);
  Node(const std::string &name, std::initializer_list<Edge *> inputs,
       std::initializer_list<Edge *> outputs);
  Node(const std::string &name, std::vector<Edge *> inputs,
       std::vector<Edge *> outputs);

  virtual ~Node();

  std::string getName();

  base::Status setDeviceType(base::DeviceType device_type);
  base::DeviceType getDeviceType();

  virtual base::Status setParam(base::Param *param);
  virtual base::Param *getParam();
  virtual base::Status setExternalParam(base::Param *external_param);

  Edge *getInput(int index = 0);
  Edge *getOutput(int index = 0);

  std::vector<Edge *> getAllInput();
  std::vector<Edge *> getAllOutput();

  bool getConstructed();

  base::Status setParallelType(const base::ParallelType &paralle_type);
  base::ParallelType getParallelType();

  void setInnerFlag(bool flag);

  void setInitializedFlag(bool flag);
  bool getInitialized();

  void setTimeProfileFlag(bool flag);
  bool getTimeProfileFlag();

  void setDebugFlag(bool flag);
  bool getDebugFlag();

  void setRunningFlag(bool flag);
  bool isRunning();

  virtual base::Status init();
  virtual base::Status deinit();

  virtual int64_t getMemorySize();
  virtual base::Status setMemory(device::Buffer *buffer);

  virtual base::EdgeUpdateFlag updataInput();

  virtual base::Status run() = 0;

 protected:
  std::string name_;
  base::DeviceType device_type_;
  std::shared_ptr<base::Param> param_;
  std::vector<base::Param *> external_param_;
  std::vector<Edge *> inputs_;
  std::vector<Edge *> outputs_;

  bool constructed_ = false;
  // 是否是图中内部节点
  bool is_inner_ = false;
  base::ParallelType parallel_type_ = base::kParallelTypeNone;
  bool initialized_ = false;
  bool is_running_ = false;
  bool is_time_profile_ = false;
  bool is_debug_ = false;
};

using SISONodeFunc =
    std::function<base::Status(Edge *input, Edge *output, base::Param *param)>;

using SIMONodeFunc = std::function<base::Status(
    Edge *input, std::initializer_list<Edge *> outputs, base::Param *param)>;

using MISONodeFunc = std::function<base::Status(
    std::initializer_list<Edge *> inputs, Edge *output, base::Param *param)>;

using MIMONodeFunc = std::function<base::Status(
    std::initializer_list<Edge *> inputs, std::initializer_list<Edge *> outputs,
    base::Param *param)>;

}  // namespace dag
}  // namespace nndeploy

#endif
