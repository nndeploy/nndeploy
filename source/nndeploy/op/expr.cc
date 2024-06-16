#include "nndeploy/op/expr.h"

namespace nndeploy {

namespace op {

std::vector<std::string> Expr::getOutputName() {
  switch (type_) {
    case kExprTypeData:
      return {data_};
    case kExprTypeOpDesc:
      return op_desc_param_->op_desc_.outputs_;
    
  }
  return {""};
}

}  // namespace op

}  // namespace nndeploy