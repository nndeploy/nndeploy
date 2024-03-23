
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
  NNOp(const std::string &name, std::vector<std::string> &model_key,
       NNOpType op_type, base::DeviceType device_type);

  virtual ~NNOp();

  // 这个接口不好，输入是模型的权重参数，用std::string表示，不够直观且不好查找权重参数
  virtual base::Status construct(std::vector<std::string> &model_value,
                                 std::vector<std::string> &model_key) = 0;

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

  virtual base::Status init(std::vector<device::Tensor *> inputs,
                            std::vector<device::Tensor *> outputs) = 0;
  virtual base::Status deinit() = 0;

  virtual base::Status reshape(std::vector<device::Tensor *> inputs) = 0;

  virtual base::Status preRun() = 0;
  virtual base::Status run() = 0;
  virtual base::Status afterRun() = 0;

 protected:
  std::string name_;
  NNOpType op_type_;
  base::DeviceType device_type_;

  std::shared_ptr<base::Param> param_;

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
