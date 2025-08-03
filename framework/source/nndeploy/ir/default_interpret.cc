#include "nndeploy/ir/default_interpret.h"

namespace nndeploy {
namespace ir {

TypeInterpretRegister<TypeInterpretCreator<DefaultInterpret>>
    g_default_interpret_register(base::kModelTypeDefault);

// DefaultInterpret::DefaultInterpret() : Interpret() {}

DefaultInterpret::DefaultInterpret(ModelDesc* model_desc, bool is_external)
    : Interpret(model_desc, is_external) {}

DefaultInterpret::~DefaultInterpret() {}

base::Status DefaultInterpret::interpret(
    const std::vector<std::string>& model_value,
    const std::vector<ValueDesc>& input) {
  base::Status status = base::kStatusCodeOk;

  // 读模型结构文件
  if (!model_value[0].empty()) {
    std::ifstream structure_stream(model_value[0],
                                   std::ifstream::in | std::ifstream::binary);
    if (!structure_stream.is_open()) {
      NNDEPLOY_LOGE("model_value[%s] is error.\n", model_value[0].c_str());
      return base::kStatusCodeErrorInvalidParam;
    }
    std::string structure_str;
    std::string line;
    while (std::getline(structure_stream, line)) {
      structure_str += line;
    }

    status = model_desc_->deserializeStructureFromJsonStr(structure_str, input);
    NNDEPLOY_RETURN_VALUE_ON_NEQ(
        status, base::kStatusCodeOk, status,
        "model_desc_->deserializeStructureFromJson failed!");
    structure_stream.close();
  }

  // 读模型权重文件
  if (model_value.size() > 1 && !model_value[1].empty()) {
    std::string warn, err;

    std::shared_ptr<safetensors::safetensors_t> mmap_st_ptr(
        new safetensors::safetensors_t());
    // 冒险的需要用愿指针
    bool ret = safetensors::mmap_from_file(model_value[1], &(*mmap_st_ptr),
                                           &warn, &err);
    if (!ret) {
      NNDEPLOY_LOGE(
          "Failed to load: %s\n"
          "  ERR: %s\n",
          model_value[1].c_str(), err.c_str());
      return base::kStatusCodeErrorIO;
    }

    // if (st_ptr_ != nullptr) {
    //   st_ptr_.reset();
    // }
    st_ptr_.emplace_back(mmap_st_ptr);
    status = model_desc_->deserializeWeightsFromSafetensors(mmap_st_ptr);
    NNDEPLOY_RETURN_VALUE_ON_NEQ(
        status, base::kStatusCodeOk, status,
        "model_desc_->deserializeWeightsFromSafetensors failed!");
  }

  return status;
}

}  // namespace ir
}  // namespace nndeploy