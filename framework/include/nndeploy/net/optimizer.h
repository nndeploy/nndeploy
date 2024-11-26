
#ifndef _NNDEPLOY_NET_OPTIMIZER_H_
#define _NNDEPLOY_NET_OPTIMIZER_H_

#include "nndeploy/ir/ir.h"
#include "nndeploy/net/util.h"
#include "nndeploy/op/op.h"

namespace nndeploy {
namespace net {

enum OptPassType : int {
  // Operator fusion
  kOptPassTypeFuseConvBias,
  kOptPassTypeFuseConvBatchNorm,
  kOptPassTypeFuseConvRelu,

  // Eliminate useless op
  kOptPassTypeEliminateCommonSubexpression,
  kOptPassTypeEliminateDeadOp,

  // Constant Folding
  kOptPassTypeFoldConstant,
};

class OptPass {
 public:
  OptPass();
  virtual ~OptPass();

  /**
   * @brief 模式匹配
   *
   * @param tensor_repository
   * @param op_repository
   * @param pass_types
   * @return op_repository中匹配到的首个op的index，如果未匹配到则返回-1
   * @note 匹配规则：
   * 1. 匹配到的op的类型为pass_types中的第一个，且op的successors_中有且只有一个
   * 2.
   * 该op的successors_的类型为pass_types中的第二个，且op的successors_的predecessors_中有且只有一个
   * 3. 以此类推，直到pass_types中的最后一个
   */
  virtual int seqPatternMatch(std::vector<TensorWrapper*>& tensor_repository,
                              std::vector<OpWrapper*>& op_repository,
                              const std::vector<ir::OpType>& types,
                              int begin_op_index);

  /**
   * @brief 模式匹配并更新tensor_repository
   *
   * @param tensor_repository
   * @param op_repository
   * @param pass_types
   * @param begin_op_index
   * @return 是否成功
   * @note 更新策略
   * 1. 更新tensor_repository：
   *    a. 更新最后一个op的输出：其生产者改为第一个op
   *    b. 删除除开最后一个op以外所有的输出
   */
  virtual base::Status seqPatternMatchUpateTensorRepository(
      std::vector<TensorWrapper*>& tensor_repository,
      std::vector<OpWrapper*>& op_repository,
      const std::vector<ir::OpType>& types, int begin_op_index);

  virtual base::Status seqPatternMatchUpateOpRepository(
      std::vector<TensorWrapper*>& tensor_repository,
      std::vector<OpWrapper*>& op_repository,
      const std::vector<ir::OpType>& types, int begin_op_index);

  /**
   * @brief 将一个Op从它前驱的后继中删除
   */
  virtual base::Status rmOpFromPredecessor(OpWrapper* op_wrapper);

  /**
   * @brief 将一个Op从它后继的前驱中删除
   */
  virtual base::Status rmOpFromSuccessors(OpWrapper* op_wrapper);

  /**
   * @brief 处理一个Op的输入Tensor
   *  将该Op从Tensor的消费者中删除，如果该Tensor的消费者仅有这一个Op作为消费者，则释放该Tensor
   */
  virtual base::Status rmInputTensorAndMaybeDelete(
      OpWrapper* op_wrapper, std::vector<TensorWrapper*>& tensor_repository);

  /**
   * @brief 处理一个Op的输出Tensor
   *  将该Op从Tensor的生产者中删除，如果该Tensor的生产者仅有这一个Op作为生产者，则释放该Tensor
   */
  virtual base::Status rmOutputTensorAndMaybeDelete(
      OpWrapper* op_wrapper, std::vector<TensorWrapper*>& tensor_repository);

  virtual base::Status optimize(std::vector<TensorWrapper*>& tensor_repository,
                                std::vector<OpWrapper*>& op_repository,
                                int begin_op_index) = 0;
};

/**
 * @brief OptPass的创建类
 *
 */
class OptPassCreator {
 public:
  virtual ~OptPassCreator(){};

  virtual std::shared_ptr<OptPass> createOptPass() = 0;
};

/**
 * @brief OptPass的创建类模板
 *
 * @tparam T
 */
template <typename T>
class TypeOptPassCreator : public OptPassCreator {
  virtual std::shared_ptr<OptPass> createOptPass() {
    return std::shared_ptr<T>(new T());
  }
};

/**
 * @brief Get the Global OptPass Creator Map object
 *
 * @return std::map<ExecutorType, std::map<const std::string &,
 * std::shared_ptr<OptPassCreator>>>&
 */
std::map<base::DeviceTypeCode,
         std::map<OptPassType, std::shared_ptr<OptPassCreator>>>&
getGlobalOptPassCreatorMap();

/**
 * @brief OptPass的创建类的注册类模板
 *
 * @tparam T
 */
template <typename T>
class TypeOptPassRegister {
 public:
  explicit TypeOptPassRegister(base::DeviceTypeCode device_type_code,
                               OptPassType type) {
    auto& creator_map = getGlobalOptPassCreatorMap();
    auto device_map = creator_map.find(device_type_code);
    if (device_map == creator_map.end()) {
      creator_map[device_type_code] =
          std::map<OptPassType, std::shared_ptr<OptPassCreator>>();
    }
    auto creator = creator_map[device_type_code].find(type);
    if (creator == creator_map[device_type_code].end()) {
      creator_map[device_type_code][type] = std::shared_ptr<T>(new T());
    }
  }
};

std::shared_ptr<OptPass> createOptPass(base::DeviceType device_type,
                                       OptPassType type);

class NNDEPLOY_CC_API Optimizer {
 public:
  Optimizer();
  ~Optimizer();

  base::Status init(base::DeviceType device_type,
                    std::set<OptPassType> enable_pass,
                    std::set<OptPassType> disable_pass);
  base::Status deinit();

  base::Status addPass(OptPassType type);
  base::Status removePass(OptPassType type);

  base::Status optimize(std::vector<TensorWrapper*>& tensor_repository,
                        std::vector<OpWrapper*>& op_repository);

 protected:
  base::DeviceType device_type_;
  std::map<OptPassType, std::shared_ptr<OptPass>> opt_passes_;
};

}  // namespace net
}  // namespace nndeploy

#endif /* _NNDEPLOY_NET_OPTIMIZER_H_ */
