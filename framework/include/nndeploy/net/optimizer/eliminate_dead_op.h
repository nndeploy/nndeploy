#ifndef _NNDEPLOY_NET_OPTIMIZER_ELIMINATE_DEAD_OP_H_
#define _NNDEPLOY_NET_OPTIMIZER_ELIMINATE_DEAD_OP_H_

#include "nndeploy/net/optimizer.h"

namespace nndeploy {

namespace net {

/**
 * 消除死节点
 * 死节点定义为 没有消费者节点的Op，且其的任意输出Tensor都不是Net的输出
 *
 */
class EliminateDeadOp : public OptPass {
 public:
  EliminateDeadOp();
  virtual ~EliminateDeadOp();

  virtual base::Status optimize(std::vector<TensorWrapper*>& tensor_repository,
                                std::vector<OpWrapper*>& op_repository,
                                int begin_op_index);
};
}  // namespace net

}  // namespace nndeploy

#endif