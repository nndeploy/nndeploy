
#ifndef _NNDEPLOY_NET_OPTIMIZER_H_
#define _NNDEPLOY_NET_OPTIMIZER_H_

#include <unordered_map>
#include <unordered_set>

#include "nndeploy/ir/ir.h"
#include "nndeploy/net/util.h"
#include "nndeploy/op/op.h"

namespace nndeploy {
namespace net {
// help types
using OpSet = std::unordered_set<ir::OpType>;

enum OptPassType : int {
  // Operator fusion
  kOptPassTypeFuseConvBias,
  kOptPassTypeFuseConvBatchNorm,
  kOptPassTypeFuseConvRelu,
  kOptPassTypeFuseConvAct,

  // Eliminate useless op
  kOptPassTypeEliminateCommonSubexpression,
  kOptPassTypeEliminateDeadOp,

  // Constant Folding
  kOptPassTypeFoldConstant,
};

class Net;

class OptPass {
 public:
  OptPass(std::string name);
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

  std::string getName();
  base::Status setNet(Net* net);

  virtual base::Status optimize(std::vector<TensorWrapper*>& tensor_repository,
                                std::vector<OpWrapper*>& op_repository,
                                int begin_op_index) = 0;

 protected:
  std::string name_;  // pass名称

  Net* net_ =
      nullptr;  //该pass所属的Net，可能要修改Net内部的数据，例如释放某些tensor
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
 *  设备类型  ->  优化等级  -> Pass类型
 */
std::map<base::DeviceTypeCode,
         std::map<int, std::map<OptPassType, std::shared_ptr<OptPassCreator>>>>&
getGlobalOptPassCreatorMap();

/**
 * @brief OptPass的创建类的注册类模板
 *
 * @tparam T
 */
template <typename T>
class TypeOptPassRegister {
 public:
  /**
   * level表示优先级，数字越小，优先级越高，在图优化时先执行，最高优先级
   * level=1； 例如FuseConvBatchNorm在FuseConvRelu之前运行
   */
  explicit TypeOptPassRegister(base::DeviceTypeCode device_type_code,
                               OptPassType type, int level) {
    auto& creator_map = getGlobalOptPassCreatorMap();
    auto device_map = creator_map.find(device_type_code);
    if (device_map == creator_map.end()) {
      creator_map[device_type_code] =
          std::map<int,
                   std::map<OptPassType, std::shared_ptr<OptPassCreator>>>();
    }
    auto level_map = creator_map[device_type_code].find(level);
    if (level_map == creator_map[device_type_code].end()) {
      creator_map[device_type_code][level] =
          std::map<OptPassType, std::shared_ptr<OptPassCreator>>();
    }

    auto creator = creator_map[device_type_code][level].find(type);
    if (creator == creator_map[device_type_code][level].end()) {
      creator_map[device_type_code][level][type] = std::shared_ptr<T>(new T());
    }
  }
};

std::shared_ptr<OptPass> createOptPass(base::DeviceType device_type, int level,
                                       OptPassType type);

class NNDEPLOY_CC_API Optimizer {
 public:
  Optimizer();
  ~Optimizer();

  base::Status init(base::DeviceType device_type,
                    std::set<OptPassType> enable_pass,
                    std::set<OptPassType> disable_pass);
  base::Status deinit();

  base::Status addPass(OptPassType type, int level);
  base::Status removePass(OptPassType type);

  base::Status optimize(std::vector<TensorWrapper*>& tensor_repository,
                        std::vector<OpWrapper*>& op_repository, Net* net);

 protected:
  base::DeviceType device_type_;
  std::map<int, std::map<OptPassType, std::shared_ptr<OptPass>>>
      opt_passes_;  // 第一个key是优先级，数字越小， 优先级越高，
                    // 在图优化时首先执行这个pass
};

}  // namespace net
}  // namespace nndeploy

#endif /* _NNDEPLOY_NET_OPTIMIZER_H_ */
