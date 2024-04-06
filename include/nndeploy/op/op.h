
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
#include "nndeploy/device/tensor.h"
#include "nndeploy/op/ir.h"

namespace nndeploy {
namespace op {

class NNDEPLOY_CC_API Op {
 public:
  Op(base::DeviceType device_type, const std::string &name, OpType op_type);

  Op(base::DeviceType device_type, const std::string &name, OpType op_type,
     std::initializer_list<const std::string &> inputs,
     std::initializer_list<const std::string &> outputs,
     std::initializer_list<const std::string &> weights = {});

  virtual ~Op();

  const std::string &getName();

  virtual base::Status setParam(base::Param *param);
  virtual base::Param *getParam();

  base::DeviceType getDeviceType();

  // 设置为virtual的原因：精度不同，内存分配不同，计算方式不同
  virtual base::Status setPrecisionType(base::PrecisionType precision_type);
  base::Status getPrecisionType();

  const std::string &getInputName(int index = 0);
  const std::string &getOutputName(int index = 0);
  const std::string &getWeightName(int index = 0);
  Tensor *getInput(int index = 0);
  Tensor *getOutput(int index = 0);
  Tensor *getWeight(int index = 0);
  base::Status setInput(Tensor *input, int index = 0);
  base::Status setOutput(Tensor *output, int index = 0);
  base::Status setWeight(Tensor *weight, int index = 0);

  std::vector<const std::string &> getAllInputName();
  std::vector<const std::string &> getAllOutputName();
  std::vector<const std::string &> getAllWeightName();
  std::vector<Tensor *> getAllInput();
  std::vector<Tensor *> getAllOutput();
  std::vector<Tensor *> getAllWeight();
  base::Status setAllInput(std::vector<Tensor *> inputs);
  base::Status setAllOutput(std::vector<Tensor *> outputs);
  base::Status setAllWeight(std::vector<Tensor *> weights);

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

  virtual base::Status reshape(std::vector<device::Tensor *> inputs);

  virtual base::Status preRun();
  virtual base::Status run() = 0;
  virtual base::Status postRun();

 protected:
  OpDesc op_desc;
  std::shared_ptr<base::Param> op_param_;

  base::DeviceType device_type_;
  base::PrecisionType precision_type_ = base::kPrecisionTypeFp32;

  std::vector<device::Tensor *> inputs_;
  std::vector<device::Tensor *> outputs_;

  std::vector<device::Tensor *> weights_;
  std::vector<device::Tensor *> variables_;

  bool constructed_ = false;
  // 是否是图中内部节点
  bool is_inner_ = false;
  base::ParallelType parallel_type_ = base::kParallelTypeNone;
  bool initialized_ = false;
  bool is_running_ = false;
  bool is_time_profile_ = false;
  bool is_debug_ = false;
};

/**
 * @brief Op的创建类
 *
 */
class OpCreator {
 public:
  virtual ~OpCreator(){};
  virtual Op *CreateOp(
      base::DeviceType device_type, const std::string &name, OpType op_type,
      std::initializer_list<const std::string &> inputs,
      std::initializer_list<const std::string &> outputs,
      std::initializer_list<const std::string &> weights = {}) = 0;
};

/**
 * @brief Op的创建类模板
 *
 * @tparam T
 */
template <typename T>
class TypeOpCreator : public OpCreator {
  virtual Op *CreateOp(
      base::DeviceType device_type, const std::string &name, OpType op_type,
      std::initializer_list<const std::string &> inputs,
      std::initializer_list<const std::string &> outputs,
      std::initializer_list<const std::string &> weights = {}) {
    return new T(device_type, name, op_type, inputs, outputs, weights);
  }
};

/**
 * @brief Get the Global Op Creator Map object
 *
 * @return std::map<ExecutorType, std::map<const std::string &,
 * std::shared_ptr<OpCreator>>>&
 */
std::map<base::DeviceType, std::map<OpType, std::shared_ptr<OpCreator>>> &
getGlobalOpCreatorMap();

/**
 * @brief Op的创建类的注册类模板
 *
 * @tparam T
 */
template <typename T>
class TypeOpRegister {
 public:
  explicit TypeOpRegister(base::DeviceType device_type, OpType op_type, ) {
    getGlobalOpCreatorMap()[device_type][op_type] = std::shared_ptr<T>(new T());
  }
};

Op *createOp(base::DeviceType device_type, const std::string &name,
             OpType op_type, std::initializer_list<const std::string &> inputs,
             std::initializer_list<const std::string &> outputs,
             std::initializer_list<const std::string &> weights = {});

using SISOOpFunc =
    std::function<base::Status(device::Tensor *input, device::Tensor *output,
                               std::shared_ptr<base::Param> op_param)>;

using SIMOOpFunc = std::function<base::Status(
    device::Tensor *input, std::initializer_list<device::Tensor *> outputs,
    std::shared_ptr<base::Param> op_param)>;

using MISOOpFunc = std::function<base::Status(
    std::initializer_list<device::Tensor *> inputs, device::Tensor *output,
    std::shared_ptr<base::Param> op_param)>;

using MIMOOpFunc =
    std::function<base::Status(std::initializer_list<device::Tensor *> inputs,
                               std::initializer_list<device::Tensor *> outputs,
                               std::shared_ptr<base::Param> op_param)>;

}  // namespace op
}  // namespace nndeploy

#endif
