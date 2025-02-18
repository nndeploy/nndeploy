#ifndef _NNDEPLOY_OP_OP_ROTATE_EMBEDDING_H_
#define _NNDEPLOY_OP_OP_ROTATE_EMBEDDING_H_

#include "nndeploy/ir/ir.h"
#include "nndeploy/op/op.h"

namespace nndeploy {

namespace op {

class OpRotateEmbedding : public Op {
 public:
  OpRotateEmbedding() : Op() {}
  virtual ~OpRotateEmbedding() {}

  // virtual base::Status inferShape();

  virtual base::Status run();

  // 重载run 函数，实现了具有不同参数情况的具体实现；
  // base::Status run();
};

NNDEPLOY_CC_API base::Status rotate_embedding(device::Tensor *input,
                                              device::Tensor *out_cos,
                                              device::Tensor *out_sin);
}  // namespace op
}  // namespace nndeploy

#endif
