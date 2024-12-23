#ifndef _NNDEPLOY_NET_OPTIMIZER_ELIMINATE_COMMON_SUBEXPRESSION_H_
#define _NNDEPLOY_NET_OPTIMIZER_ELIMINATE_COMMON_SUBEXPRESSION_H_

#include "nndeploy/net/optimizer.h"

namespace nndeploy {

namespace net {

/**
 * 消除公共子表达式
 *
 */
class EliminateCommonSubexpression : public OptPass {
 public:
  EliminateCommonSubexpression();
  virtual ~EliminateCommonSubexpression();

  virtual base::Status optimize(std::vector<TensorWrapper*>& tensor_repository,
                                std::vector<OpWrapper*>& op_repository,
                                int begin_op_index);
};

}  // namespace net

}  // namespace nndeploy

#endif