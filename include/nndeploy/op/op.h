
#ifndef _NNDEPLOY_OP_OP_H_
#define _NNDEPLOY_OP_OP_H_

#include "nndeploy/base/common.h"
#include "nndeploy/base/glic_stl_include.h"
#include "nndeploy/base/log.h"
#include "nndeploy/base/macro.h"
#include "nndeploy/base/object.h"
#include "nndeploy/base/param.h"
#include "nndeploy/base/status.h"
#include "nndeploy/base/string.h"
#include "nndeploy/base/time_profiler.h"
#include "nndeploy/base/value.h"
#include "nndeploy/device/buffer.h"
#include "nndeploy/device/buffer_pool.h"
#include "nndeploy/device/device.h"
#include "nndeploy/device/mat.h"
#include "nndeploy/device/tensor.h"
#include "nndeploy/op/type.h"

namespace nndeploy {
namespace op {

class NNDEPLOY_CC_API NNOp {
 public:
  /**
   * @brief Construct a new NNOp object
   *
   * @param name - op name
   * @param op_type - op type
   * @param device_type - device type
   */
  NNOp(const std::string &name, NNOpType op_type, base::DeviceType device_type,
       interpreter::Interpreter *interpreter,
       std::vector<std::string> &weight_key,
       std::vector<device::Tensor *> inputs,
       std::vector<device::Tensor *> outputs);

  virtual ~NNOp();

  std::string getName();

  base::DeviceType getDeviceType();

  virtual base::Status setParam(base::Param *param);
  virtual base::Param *getParam();

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

  virtual base::Status init() = 0;
  virtual base::Status deinit() = 0;

  virtual base::Status reshape(std::vector<device::Tensor *> inputs) = 0;

  virtual base::Status run() = 0;

 protected:
  std::string name_;
  NNOpType op_type_;
  base::DeviceType device_type_;

  std::shared_ptr<base::Param> param_;

  interpreter::Interpreter *interpreter_;
  std::vector<std::string> weight_key_;

  std::vector<device::Tensor *> inputs_;
  std::vector<device::Tensor *> outputs_;

  bool constructed_ = false;
  // 是否是图中内部节点
  bool is_inner_ = false;
  base::ParallelType parallel_type_ = base::kParallelTypeNone;
  bool initialized_ = false;
  bool is_running_ = false;
  bool is_time_profile_ = false;
  bool is_debug_ = false;
};

using SingleIONNOpFunc = std::function<base::Status(
    device::Tensor *input, device::Tensor *output, base::Param *param)>;

using MultiIONNOpFunc = std::function<base::Status(
    std::initializer_list<device::Tensor *> input,
    std::initializer_list<device::Tensor *> output, base::Param *param)>;

}  // namespace op
}  // namespace nndeploy

#endif
