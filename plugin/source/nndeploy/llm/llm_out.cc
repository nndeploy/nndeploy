#include "nndeploy/llm/llm_out.h"

namespace nndeploy {
namespace llm {

base::Status LlmOut::serialize(rapidjson::Value& json,
                               rapidjson::Document::AllocatorType& allocator) {
  this->addRequiredParam("path_");
  base::Status status = dag::Node::serialize(json, allocator);
  if (status != base::kStatusCodeOk) {
    return status;
  }
  rapidjson::Value path_val;
  path_val.SetString(path_.c_str(),
                     static_cast<rapidjson::SizeType>(path_.length()),
                     allocator);
  json.AddMember("path_", path_val, allocator);
  return base::kStatusCodeOk;
}

base::Status LlmOut::deserialize(rapidjson::Value& json) {
  base::Status status = dag::Node::deserialize(json);
  if (status != base::kStatusCodeOk) {
    return status;
  }
  if (json.HasMember("path_") && json["path_"].IsString()) {
    path_ = json["path_"].GetString();
  }

  return base::kStatusCodeOk;
}

base::Status LlmOut::run() {
  tokenizer::TokenizerText* result =
      (tokenizer::TokenizerText*)(inputs_[0]->getParam(this));
  if (result == nullptr) {
    return base::kStatusCodeErrorInvalidValue;
  }

  if (!path_.empty()) {
    std::ofstream ofs(path_.c_str());
    if (!ofs) {
      NNDEPLOY_LOGE("[LlmOut] Failed to open file[%s]\n", path_.c_str());
      return base::kStatusCodeErrorIO;
    }
    ofs << result->texts_[0];
    ofs.close();
  }

  return base::kStatusCodeOk;
}

REGISTER_NODE("nndeploy::llm::LlmOut", LlmOut);

}  // namespace llm
}  // namespace nndeploy
