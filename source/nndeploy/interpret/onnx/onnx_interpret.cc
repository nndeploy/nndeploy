

#include "nndeploy/interpret/onnx/onnx_interpret.h"

#include <google/protobuf/message.h>
#include <google/protobuf/text_format.h>

#include "google/protobuf/io/coded_stream.h"
#include "google/protobuf/io/zero_copy_stream_impl.h"
#include "nndeploy/op/ir.h"
#include "nndeploy/op/op.h"

namespace nndeploy {
namespace interpret {

base::Status OnnxInterpret::interpret(
    const std::vector<std::string> &model_value,
    const std::vector<op::ValueDesc> &input) {
  base::Status status = base::kStatusCodeOk;

  // 解释模型
  std::ifstream input_stream(model_value[0],
                             std::ifstream::in | std::ifstream::binary);
  if (!input_stream.is_open()) {
    NNDEPLOY_LOGE("model_value[%s] is error.\n", model_value[0].c_str());
    return base::kStatusCodeErrorInvalidParam;
  }
  google::protobuf::io::IstreamInputStream proto_input_stream(&input_stream);
  google::protobuf::io::CodedInputStream coded_input_stream(
      &proto_input_stream);
#if GOOGLE_PROTOBUF_VERSION >= 3002000
  coded_input_stream.SetTotalBytesLimit(INT_MAX);
#else
  coded_input_stream.SetTotalBytesLimit(INT_MAX, INT_MAX / 2);
#endif
  this->onnx_model_ = std::unique_ptr<onnx::ModelProto>(new onnx::ModelProto());
  bool success = this->onnx_model_->ParseFromCodedStream(&coded_input_stream);
  input_stream.close();

  return status;
}

}  // namespace interpret
}  // namespace nndeploy
