
#ifndef _NNDEPLOY_INTERPRET_INTERPRET_H_
#define _NNDEPLOY_INTERPRET_INTERPRET_H_

#include "nndeploy/op/ir.h"
#include "nndeploy/op/op.h"

namespace nndeploy {
namespace interpret {

class Interpret {
 public:
  Interpret() : model_desc_(nullptr){};
  virtual ~Interpret() {
    if (model_desc_ != nullptr) {
      delete model_desc_;
    }
  };

  virtual base::Status interpret(
      const std::vector<std::string> &model_value,
      const std::vector<op::ValueDesc> &input = {}) = 0;

  op::ModelDesc *getModelDesc() { return model_desc_; };

 protected:
  op::ModelDesc *model_desc_;
};

}  // namespace interpret
}  // namespace nndeploy

#endif /* _NNDEPLOY_NET_INTERPRET_H_ */