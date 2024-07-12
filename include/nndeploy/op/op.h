
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
#include "nndeploy/device/device.h"
#include "nndeploy/device/memory_pool.h"
#include "nndeploy/device/tensor.h"
#include "nndeploy/op/ir.h"
// #include "nndeploy/op/model_desc.h"

namespace nndeploy {
namespace op {

class NNDEPLOY_CC_API Op {
 public:
  Op();

  virtual ~Op();

  base::Status setName(std::string name);
  std::string getName();

  virtual base::Status setParam(std::shared_ptr<base::Param> param);
  virtual std::shared_ptr<base::Param> getParam();

  base::Status setDeviceType(base::DeviceType device_type);
  base::DeviceType getDeviceType();

  base::Status setOpType(OpType op_type);
  OpType getOpType();

  // 设置为virtual的原因：精度不同，内存分配不同，计算方式不同
  virtual base::Status setPrecisionType(base::PrecisionType precision_type);
  base::Status getPrecisionType();

  std::string getInputName(int index = 0);
  std::string getOutputName(int index = 0);

  device::Tensor *getInput(int index = 0);
  device::Tensor *getOutput(int index = 0);

  virtual base::Status setInput(device::Tensor *input);
  virtual base::Status setOutput(device::Tensor *output);

  base::Status setInput(device::Tensor *input, int index);
  base::Status setOutput(device::Tensor *output, int index);

  base::Status setAllInputName(std::initializer_list<std::string>);
  base::Status setAllOutputName(std::initializer_list<std::string>);

  base::Status setAllInputName(std::vector<std::string> &);
  base::Status setAllOutputName(std::vector<std::string> &);

  std::vector<std::string> getAllInputName();
  std::vector<std::string> getAllOutputName();

  std::vector<device::Tensor *> getAllInput();
  std::vector<device::Tensor *> getAllOutput();

  base::Status setAllInput(std::vector<device::Tensor *> inputs);
  base::Status setAllOutput(std::vector<device::Tensor *> outputs);

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

  virtual base::Status reshape(base::ShapeMap &shape_map);
  virtual base::Status inferDataType(
      std::map<std::string, base::DataType> &dtype_map);

  virtual base::Status preRun();
  virtual base::Status run() = 0;
  virtual base::Status postRun();

 protected:
  OpDesc op_desc_;

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

  virtual Op *createOp(base::DeviceType device_type, const std::string &name,
                       OpType op_type) = 0;

  virtual Op *createOp(base::DeviceType device_type, const std::string &name,
                       OpType op_type,
                       std::initializer_list<std::string> inputs,
                       std::initializer_list<std::string> outputs) = 0;

  virtual Op *createOp(base::DeviceType device_type, const std::string &name,
                       OpType op_type, std::vector<std::string> &inputs,
                       std::vector<std::string> &outputs) = 0;
};

/**
 * @brief Op的创建类模板
 *
 * @tparam T
 */
template <typename T>
class TypeOpCreator : public OpCreator {
  virtual Op *createOp(base::DeviceType device_type, const std::string &name,
                       OpType op_type) {
    auto op = new T();
    op->setDeviceType(device_type);
    op->setName(name);
    op->setOpType(op_type);
    return op;
  }

  virtual Op *createOp(base::DeviceType device_type, const std::string &name,
                       OpType op_type,
                       std::initializer_list<std::string> inputs,
                       std::initializer_list<std::string> outputs) {
    auto op = new T();
    op->setDeviceType(device_type);
    op->setName(name);
    op->setOpType(op_type);
    op->setAllInputName(inputs);
    op->setAllOutputName(outputs);
    return op;
  }

  virtual Op *createOp(base::DeviceType device_type, const std::string &name,
                       OpType op_type, std::vector<std::string> &inputs,
                       std::vector<std::string> &outputs) {
    auto op = new T();
    op->setDeviceType(device_type);
    op->setName(name);
    op->setOpType(op_type);
    op->setAllInputName(inputs);
    op->setAllOutputName(outputs);
    return op;
  }
};

/**
 * @brief Get the Global Op Creator Map object
 *
 * @return std::map<ExecutorType, std::map<const std::string &,
 * std::shared_ptr<OpCreator>>>&
 */
std::map<base::DeviceTypeCode, std::map<OpType, std::shared_ptr<OpCreator>>> &
getGlobalOpCreatorMap();

/**
 * @brief Op的创建类的注册类模板
 *
 * @tparam T
 */
template <typename T>
class TypeOpRegister {
 public:
  explicit TypeOpRegister(base::DeviceTypeCode device_type_code,
                          OpType op_type) {
    getGlobalOpCreatorMap()[device_type_code][op_type] =
        std::shared_ptr<T>(new T());
  }
};

Op *createOp(base::DeviceType device_type, const std::string &name,
             OpType op_type);

Op *createOp(base::DeviceType device_type, const std::string &name,
             OpType op_type, std::initializer_list<std::string> inputs,
             std::initializer_list<std::string> outputs);

Op *createOp(base::DeviceType device_type, const std::string &name,
             OpType op_type, std::vector<std::string> &inputs,
             std::vector<std::string> &outputs);

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

#define REGISTER_OP_IMPLEMENTION(device_type_code, op_type, op_class) \
  TypeOpRegister<TypeOpCreator<op_class>> g_##op_class##_register(    \
      device_type_code, op_type);

}  // namespace op
}  // namespace nndeploy

#endif
