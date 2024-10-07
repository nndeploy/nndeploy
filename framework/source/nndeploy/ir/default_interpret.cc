#include "nndeploy/ir/default_interpret.h"

namespace nndeploy {
namespace ir {

DefaultInterpret::DefaultInterpret() : Interpret() {}
DefaultInterpret::~DefaultInterpret() {}

base::Status DefaultInterpret::interpret(
    const std::vector<std::string>& model_value,
    const std::vector<ValueDesc>& input) {
  base::Status status = base::kStatusCodeOk;

  // 读模型结构文件
  std::ifstream structure_stream(model_value[0],
                                 std::ifstream::in | std::ifstream::binary);
  if (!structure_stream.is_open()) {
    NNDEPLOY_LOGE("model_value[%s] is error.\n", model_value[0].c_str());
    return base::kStatusCodeErrorInvalidParam;
  }
  status = model_desc_->deserializeStructureFromText(structure_stream, input);
  NNDEPLOY_RETURN_VALUE_ON_NEQ(
      status, base::kStatusCodeOk, status,
      "model_desc_->deserializeStructureFromText failed!");

  // 读模型结构文件
  std::ifstream weight_stream(model_value[1],
                              std::ifstream::in | std::ifstream::binary);
  if (!weight_stream.is_open()) {
    NNDEPLOY_LOGE("model_value[%s] is error.\n", model_value[0].c_str());
    return base::kStatusCodeErrorInvalidParam;
  }

  status = model_desc_->deserializeWeightsFromBinary(weight_stream);

  NNDEPLOY_RETURN_VALUE_ON_NEQ(
      status, base::kStatusCodeOk, status,
      "model_desc_->deserializeWeightsFromBinary failed!");

  return status;
}

}  // namespace ir
}  // namespace nndeploy