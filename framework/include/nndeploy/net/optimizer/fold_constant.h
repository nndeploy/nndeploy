#ifndef _NNDEPLOY_NET_OPTIMIZER_FOLD_CONSTANT_H_
#define _NNDEPLOY_NET_OPTIMIZER_FOLD_CONSTANT_H_

#include "nndeploy/net/optimizer.h"

namespace nndeploy {

namespace net {

/**
 * 静态常量折叠
 * 检测编译器可以运行的Op，即所有输入都是常量
 */
class FoldConstant : public OptPass {
 public:
  FoldConstant();
  virtual ~FoldConstant();

  virtual base::Status optimize(std::vector<TensorWrapper*>& tensor_repository,
                                std::vector<OpWrapper*>& op_repository,
                                int begin_op_index);
  bool isDeterministic(const OpWrapper* op_wrapper);

  bool isQDQ(const OpWrapper* op_wrapper);

  bool isAllInputConstant(const OpWrapper* op_wrapper,
                          const std::vector<TensorWrapper*>& tensor_repository);

  bool produceLargeTensor(const OpWrapper* op_wrapper,
                          const std::vector<TensorWrapper*>& tensor_repository);

  bool isSupportFold(const OpWrapper* op_wrapper);

  base::Status runOp(const OpWrapper* op_wrapper);
};
}  // namespace net
}  // namespace nndeploy

#endif