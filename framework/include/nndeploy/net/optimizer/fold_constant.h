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
};
}  // namespace net
}  // namespace nndeploy

#endif