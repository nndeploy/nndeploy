
#ifndef _NNDEPLOY_OP_OP_EMBEDDING_H_
#define _NNDEPLOY_OP_OP_EMBEDDING_H_

#include "nndeploy/ir/ir.h"
#include "nndeploy/op/op.h"

namespace nndeploy {
namespace op {

class OpEmbedding : public Op {
 public:
  OpEmbedding() : Op() {}
  virtual ~OpEmbedding() {}

  virtual base::Status inferShape();

  virtual base::Status run();
};

NNDEPLOY_CC_API base::Status embedding(device::Tensor* input,
                                    device::Tensor* indice,
                                    device::Tensor* output);

}  // namespace op
}  // namespace nndeploy

#endif
