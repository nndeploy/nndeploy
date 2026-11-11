/*
 *   Copyright (c) 2025
 *   All rights reserved.
 */
#ifndef _NNDEPLOY_OP_OP_CAST_H_
#define _NNDEPLOY_OP_OP_CAST_H_

#include "nndeploy/ir/ir.h"
#include "nndeploy/op/op.h"
#include "nndeploy/op/op_unary.h"

namespace nndeploy {
namespace op {

class OpCast : public OpUnary {
 public:
  OpCast() : OpUnary() { is_inplace_ = false; }
  virtual ~OpCast() {}

  base::Status inferDataType() override;
  base::Status run() override;
};

NNDEPLOY_CC_API base::Status cast(device::Tensor* input,
                                  std::shared_ptr<ir::CastParam> param,
                                  device::Tensor* output);

}  // namespace op
}  // namespace nndeploy

#endif  // _NNDEPLOY_OP_OP_CAST_H_