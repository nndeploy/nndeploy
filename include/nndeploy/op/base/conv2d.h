#ifndef _NNDEPLOY_OP_BASE_CONV2D_H_
#define _NNDEPLOY_OP_BASE_CONV2D_H_

#include "nndeploy/base/param.h"
#include "nndeploy/op/op.h"

namespace nndeploy {

namespace op {

class Conv2dParam : public base::Param {
  public:
  std::vector<int> strides{1, 1};
  std::vector<int> padding{0, 0};
  std::vector<int> kernel_size;
};

class BaseConv2dOp : public Op  {

};

}  // namespace op
}  // namespace nndeploy

#endif