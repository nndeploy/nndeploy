
#ifndef _NNDEPLOY_NET_NET_H_
#define _NNDEPLOY_NET_NET_H_

#include "nndeploy/net/session.h"
#include "nndeploy/net/util.h"
#include "nndeploy/op/ir.h"
#include "nndeploy/op/op.h"

namespace nndeploy {
namespace net {

class NNDEPLOY_CC_API Net : public op::Op {
 public:
  Net();
  virtual ~Net();

  // 在这个函数之前调用setDeviceType
  base::Status setModelDesc(std::shared_ptr<op::ModelDesc> model_desc);

  TensorWrapper *createTensor(const std::string &name);
  TensorWrapper *addTensor(device::Tensor *tensor, bool is_external = true);
  device::Tensor *getTensor(const std::string &name);

  bool isWeight(const std::string &name);
  base::Status covertWeight(op::Op *op, const std::string &weight);

  op::Op *createOp(base::DeviceType device_type, const std::string &name,
                   op::OpType op_type,
                   std::initializer_list<std::string> inputs,
                   std::initializer_list<std::string> outputs);
  op::Op *createOp(base::DeviceType device_type, const std::string &name,
                   op::OpType op_type, std::vector<std::string> &inputs,
                   std::vector<std::string> &outputs);
  base::Status addNet(Net *net, bool is_external);

  base::Status setOpParam(const std::string &op_name,
                          std::shared_ptr<base::Param> param);
  std::shared_ptr<base::Param> getOpParam(const std::string &op_name);

  virtual base::Status init();
  virtual base::Status deinit();

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
  std::shared_ptr<op::ModelDesc> model_desc_;

  std::vector<TensorWrapper *> tensor_repository_;
  /*
   * 设备 - 例如cann而言，就需要吧model_desc_中的权重文件加载到cann中
   */
  std::map<std::string, device::Tensor *> weights_;
  std::vector<OpWrapper *> op_repository_;

  Session *session_;
};

Net *createNet(std::shared_ptr<op::ModelDesc> model_desc,
               base::DeviceType device_type,
               base::PrecisionType precision_type);

}  // namespace net
}  // namespace nndeploy

#endif