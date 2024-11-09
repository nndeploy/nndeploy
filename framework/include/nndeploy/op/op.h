
#ifndef _NNDEPLOY_OP_OP_H_
#define _NNDEPLOY_OP_OP_H_

#include "nndeploy/base/common.h"
#include "nndeploy/base/glic_stl_include.h"
#include "nndeploy/base/log.h"
#include "nndeploy/base/macro.h"
#include "nndeploy/base/object.h"
#include "nndeploy/base/param.h"
#include "nndeploy/base/shape.h"
#include "nndeploy/base/status.h"
#include "nndeploy/base/string.h"
#include "nndeploy/base/time_profiler.h"
#include "nndeploy/base/any.h"
#include "nndeploy/device/buffer.h"
#include "nndeploy/device/device.h"
#include "nndeploy/device/memory_pool.h"
#include "nndeploy/device/tensor.h"
#include "nndeploy/ir/ir.h"

namespace nndeploy {
namespace op {

/**
 * @brief Op的基类
 * @note
 * # 单算子模式
 * ## 当输出tensor为空时，内部分配
 * ## 当输出tensor不为空时，检测当前输出tensor的内存是否足够，如果足够，则直接使用，否则报错
 * # 计算图Net模式
 * ## 静态shape，由tensor pool分配
 * ## 动态shape
 * ### 指定了max_shape，则由tensor pool按最大shape分配，调用reshape函数时，只是重新调整了tensor逻辑shape的大小
 * ### 未指定max_shape，每次调用reshape函数时，在计算图层面都会：先释放上一次分配的内存，再重新分配内存
 * ## 大语言模型
 * ### kvblock的方式
 */
class NNDEPLOY_CC_API Op {
 public:
  Op();

  virtual ~Op();

  base::Status setName(std::string name);
  std::string getName();

  base::Status setOpType(ir::OpType op_type);
  ir::OpType getOpType();

  virtual base::Status setParam(std::shared_ptr<base::Param> param);
  virtual std::shared_ptr<base::Param> getParam();

  base::Status setDeviceType(base::DeviceType device_type);
  base::DeviceType getDeviceType();

  /**
   * @brief 设置精度类型 精度不同，计算方式不同，内存分配不同
   *
   * @param precision_type
   * @return base::Status
   * @note
   * 当且仅当data_type为浮点数类型时，precision_type_会与data_type一起，共同决定具体调用的kernel函数
   */
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

  /**
   * @brief 类型推理
   *
   * @return base::Status
   * @note 当输入的data_type确定时，在计算图Net::init中会调用该函数
   */
  virtual base::Status inferDataType();
  /**
   * @brief 形状推理
   *
   * @return base::Status
   * @note 当输入的shape确定时，在计算图Net::init中调用该函数
   */
  virtual base::Status inferShape();
  /**
   * @brief 数据格式推理
   *
   * @return base::Status
   * @note 当输入的shape数据格式时，在计算图Net::init中调用该函数
   */
  virtual base::Status inferDataFormat();

  /**
   * @brief 初始化
   *
   * @return base::Status
   * @note 功能
   * 1. 参数
   * 2. 权重
   */
  virtual base::Status init();
  virtual base::Status deinit();

  /**
   * @brief 重新推理形状，通常在初始化之后、preRun前调用
   *
   * @param shape_map
   * @return base::Status
   */
  virtual base::Status reshape(base::ShapeMap &shape_map);

  /**
   * @brief preRun 确定输入后运行，只运行一次
   *
   * @return base::Status
   * @note
   * 当输入、参数没修改时，该函数只运行一次。
   * # 检查输出是否需要重新分配内存
   * # 确定workspace
   */
  virtual base::Status preRun();
  /**
   * @brief 得到op的workspace大小
   * note: op在运行时的workspace大小，在输入确定后调用
   * eg：例如Conv，当存在padding时，需要分配额外的内存，存放padding后的内存
   */
  virtual uint64_t getWorkspaceSize();
  virtual void setWorkspace(void *workspace);
  /**
   * @brief 得到op的flops
   *
   * @return uint64_t
   */
  virtual uint64_t getFlops();

  virtual base::Status run() = 0;
  virtual base::Status postRun();

 protected:
  /**
   * @brief op的描述
   * 包含op的类型、名称、输入名称、输出名称、参数
   */
  ir::OpDesc op_desc_;

  /**
   * @brief op的设备类型
   */
  base::DeviceType device_type_;
  /**
   * @brief op的精度类型
   * note: 精度类型与输入输出tensor的data_type的不同
   * # data_type大部分时候决定具体调用的kernel函数
   * #
   * 当且仅当data_type为浮点数类型时，precision_type_会与data_type一起，共同决定具体调用的kernel函数
   */
  base::PrecisionType precision_type_ = base::kPrecisionTypeFp32;

  /**
   * @brief op的输入tensor
   * note: 当权重为tensor时，权重tensor也会在这里
   * eg:
   * # 当op为Conv时，inputs_[0]为输入数据，inputs_[1]为weight, inputs_[2]为bias
   * 内存分配
   * # 权重内存：
   * ## 在初始化时完成，当权重与推理设备要求的权重一致时，浅拷贝即可
   * ## 当权重与推理设备要求的权重不一致时，需要进行内存迁移
   * # op输入
   * ## 已经完成分配
   */
  std::vector<device::Tensor *> inputs_;
  /**
   * @brief op的输出tensor
   * 
   * @note: outputs_的内存分配
   * # 单算子模式
   * ## 当输出tensor为空时，内部分配
   * ## 当输出tensor不为空时，检测当前输出tensor的内存是否足够，如果足够，则直接使用，否则报错
   * # 计算图Net模式
   * ## 静态shape，由tensor pool分配
   * ## 动态shape
   * ### 指定了max_shape，则由tensor pool按最大shape分配，调用reshape函数时，只是重新调整了tensor逻辑shape的大小
   * ### 未指定max_shape，每次调用reshape函数时，在计算图层面都会：先释放上一次分配的内存，再重新分配内存
   * ## 大语言模型
   * ### kvblock的方式
   */
  std::vector<device::Tensor *> outputs_;

  /**
   * @brief op的workspace大小
   * note: op在运行时的workspace大小, 在preRun中确定
   * eg：例如Conv，当存在padding时，需要分配额外的内存，存放padding后的内存
   */
  bool workspace_is_external_ = false;  // workspace是否是外部传入
  uint64_t workspace_size_ = 0;         // workspace大小
  void *workspace_ = nullptr;           // op的workspace
  uint64_t flops_ = 0;                  // op的flops

  // 是否是图中内部节点
  bool is_inner_ = false;
  // 并行类型
  base::ParallelType parallel_type_ = base::kParallelTypeNone;
  // 是否 可以是inplace op
  bool is_inplace_ = false;
  // 参数&输入是否发生变化
  bool is_changed_ = false;
  
  bool constructed_ = false;
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
  virtual ~OpCreator() {};

  virtual Op *createOp(base::DeviceType device_type, const std::string &name,
                       ir::OpType op_type) = 0;

  virtual Op *createOp(base::DeviceType device_type, const std::string &name,
                       ir::OpType op_type,
                       std::initializer_list<std::string> inputs,
                       std::initializer_list<std::string> outputs) = 0;

  virtual Op *createOp(base::DeviceType device_type, const std::string &name,
                       ir::OpType op_type, std::vector<std::string> &inputs,
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
                       ir::OpType op_type) {
    auto op = new T();
    op->setDeviceType(device_type);
    op->setName(name);
    op->setOpType(op_type);
    return op;
  }

  virtual Op *createOp(base::DeviceType device_type, const std::string &name,
                       ir::OpType op_type,
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
                       ir::OpType op_type, std::vector<std::string> &inputs,
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
std::map<base::DeviceTypeCode,
         std::map<ir::OpType, std::shared_ptr<OpCreator>>> &
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
                          ir::OpType op_type) {
    getGlobalOpCreatorMap()[device_type_code][op_type] =
        std::shared_ptr<T>(new T());
  }
};

Op *createOp(base::DeviceType device_type, const std::string &name,
             ir::OpType op_type);

Op *createOp(base::DeviceType device_type, const std::string &name,
             ir::OpType op_type, std::initializer_list<std::string> inputs,
             std::initializer_list<std::string> outputs);

Op *createOp(base::DeviceType device_type, const std::string &name,
             ir::OpType op_type, std::vector<std::string> &inputs,
             std::vector<std::string> &outputs);

Op *createOp(base::DeviceType device_type, const std::string &name,
             ir::OpType op_type, std::vector<std::string> &inputs,
             std::vector<std::string> &outputs,
             std::shared_ptr<base::Param> param);

Op *createOp(base::DeviceType device_type, std::shared_ptr<ir::OpDesc> op_desc);

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
