
#ifndef _NNDEPLOY_NET_NET_H_
#define _NNDEPLOY_NET_NET_H_

#include "nndeploy/ir/ir.h"
#include "nndeploy/net/session.h"
#include "nndeploy/net/util.h"
#include "nndeploy/op/op.h"

namespace nndeploy {
namespace net {

class NNDEPLOY_CC_API Net : public op::Op {
 public:
  Net();
  virtual ~Net();

  // 在这个函数之前调用setDeviceType
  base::Status setModelDesc(ir::ModelDesc *model_desc);
  base::Status setDynamicShape(bool is_dynamic_shape, base::ShapeMap &min_shape,
                               base::ShapeMap &opt_shape,
                               base::ShapeMap &max_shape);
  base::Status setTensorPoolType(TensorPoolType tensor_pool_type);

  TensorWrapper *createTensor(const std::string &name, bool is_weight = false);
  TensorWrapper *addTensor(device::Tensor *tensor, bool is_external = true,
                           bool is_weight = false);
  device::Tensor *getTensor(const std::string &name);

  bool isWeight(const std::string &name);
  // 有转移所有权属性
  device::Tensor *getWeight(const std::string &weight);

  op::Op *createOp(base::DeviceType device_type, const std::string &name,
                   ir::OpType op_type,
                   std::initializer_list<std::string> inputs,
                   std::initializer_list<std::string> outputs);
  op::Op *createOp(base::DeviceType device_type, const std::string &name,
                   ir::OpType op_type, std::vector<std::string> &inputs,
                   std::vector<std::string> &outputs);
  base::Status addNet(Net *net, bool is_external);

  base::Status setOpParam(const std::string &op_name,
                          std::shared_ptr<base::Param> param);
  std::shared_ptr<base::Param> getOpParam(const std::string &op_name);

  virtual base::Status init();
  virtual base::Status deinit();

  /**
   * @brief 获取推理所需的内存大小
   *
   * @return int64_t
   */
  virtual int64_t getMemorySize();
  /**
   * @brief 设置推理所需的内存（推理内存由外部分配）
   *
   * @param buffer
   * @return base::Status
   */
  virtual base::Status setMemory(device::Buffer *buffer);

  virtual base::Status inferDataType();
  virtual base::Status inferShape();
  virtual base::Status inferDataFormat();

  virtual base::Status reshape(base::ShapeMap &shape_map);

  virtual base::Status preRun();
  virtual base::Status run();
  virtual base::Status postRun();

  base::Status dump(std::ostream &oss);

 protected:
  virtual base::Status construct();
  // NNDEPLOY_LOGI("1. Optimizer Graph V1!\n");
  base::Status optimizer();
  // NNDEPLOY_LOGI("##############\n");
  // NNDEPLOY_LOGI("session init\n");
  // NNDEPLOY_LOGI("#. Optimizer Graph V2!\n");
  // NNDEPLOY_LOGI("#. Memory Allocation Phase!\n");
  // NNDEPLOY_LOGI("#. Cost Calculations!\n");
  // NNDEPLOY_LOGI("##############\n");
  virtual base::Status session();

 protected:
  ir::ModelDesc *model_desc_;

  std::vector<TensorWrapper *> tensor_repository_;
  std::vector<OpWrapper *> op_repository_;

  bool is_dynamic_shape_ = false;                // 是否是动态shape
  base::ShapeMap min_shape_ = base::ShapeMap();  // 当为动态输入时最小shape
  base::ShapeMap opt_shape_ = base::ShapeMap();  // 当为动态输入时最优shape
  base::ShapeMap max_shape_ = base::ShapeMap();  // 当为动态输入时最大shape
  TensorPoolType tensor_pool_type_ =
      kTensorPool1DSharedObjectTypeGreedyBySizeImprove;

  Session *session_;
};

Net *createNet(ir::ModelDesc *model_desc, base::DeviceType device_type,
               base::PrecisionType precision_type);

}  // namespace net
}  // namespace nndeploy

#endif