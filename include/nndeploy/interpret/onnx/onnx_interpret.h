#ifndef _NNDEPLOY_INTERPRET_ONNX_INTERPRET_H_
#define _NNDEPLOY_INTERPRET_ONNX_INTERPRET_H_

#include "nndeploy/interpret/interpret.h"
#include "nndeploy/op/ir.h"
#include "nndeploy/op/op.h"
#include "onnx/onnx_pb.h"

namespace nndeploy {
namespace interpret {

class OnnxInterpret : public Interpret {
 public:
  OnnxInterpret() : Interpret(){};
  virtual ~OnnxInterpret(){};

  virtual base::Status interpret(const std::vector<std::string> &model_value,
                                 const std::vector<op::ValueDesc> &input);

 private:
  std::unique_ptr<onnx::ModelProto> onnx_model_;
};

}  // namespace interpret
}  // namespace nndeploy

#endif /* _NNDEPLOY_NET_ONNX_INTERPRET_H_ */
