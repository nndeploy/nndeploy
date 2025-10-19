#include "nndeploy/tokenizer/tokenizer_mnn/tokenizer_mnn.h"

namespace nndeploy {
namespace tokenizer {

// @ZhaodeWang:
// 继承TokenizerEncode类，其输入输出必须是TokenizerText和TokenizerIds
TokenizerEncodeMnn::TokenizerEncodeMnn(const std::string& name)
    : TokenizerEncode(name) {
  key_ = "nndeploy::tokenizer::TokenizerEncodeMnn";
  desc_ =
      "A tokenizer encode node that  to "
      "encode text into token IDs. Supports HuggingFace and BPE tokenizers. "
      "Can encode single strings or batches of text. Provides vocabulary "
      "lookup and token-to-ID conversion.";
  param_ = std::make_shared<TokenizerPraram>();
  // this->setInputTypeInfo<TokenizerText>();
  // this->setOutputTypeInfo<TokenizerIds>();
}
TokenizerEncodeMnn::TokenizerEncodeMnn(const std::string& name,
                                       std::vector<dag::Edge*> inputs,
                                       std::vector<dag::Edge*> outputs)
    : TokenizerEncode(name, inputs, outputs) {
  key_ = "nndeploy::tokenizer::TokenizerEncodeMnn";
  desc_ =
      "A tokenizer encode node that  to "
      "encode text into token IDs. Supports HuggingFace and BPE tokenizers. "
      "Can encode single strings or batches of text. Provides vocabulary "
      "lookup and token-to-ID conversion.";
  param_ = std::make_shared<TokenizerPraram>();
  // this->setInputTypeInfo<TokenizerText>();
  // this->setOutputTypeInfo<TokenizerIds>();
}

// TokenizerEncodeMnn 实现
TokenizerEncodeMnn::~TokenizerEncodeMnn() {
  // TODO: 清理资源
}

base::Status TokenizerEncodeMnn::init() {
  // TODO: 初始化MNN tokenizer
  return base::kStatusCodeOk;
}

base::Status TokenizerEncodeMnn::deinit() {
  // TODO: 反初始化MNN tokenizer
  return base::kStatusCodeOk;
}

base::Status TokenizerEncodeMnn::run() {
  // TODO: 实现运行逻辑
  return base::kStatusCodeOk;
}

std::vector<int32_t> TokenizerEncodeMnn::encode(const std::string& text) {
  // TODO: 实现文本编码
  return std::vector<int32_t>();
}

std::vector<std::vector<int32_t>> TokenizerEncodeMnn::encodeBatch(
    const std::vector<std::string>& texts) {
  // TODO: 实现批量文本编码
  return std::vector<std::vector<int32_t>>();
}

size_t TokenizerEncodeMnn::getVocabSize() {
  // TODO: 返回词汇表大小
  return 0;
}

int32_t TokenizerEncodeMnn::tokenToId(const std::string& token) {
  // TODO: 将token转换为ID
  return -1;
}

base::Status TokenizerEncodeMnn::serialize(
    rapidjson::Value& json, rapidjson::Document::AllocatorType& allocator) {
  base::Status status = dag::Node::serialize(json, allocator);
  if (status != base::kStatusCodeOk) {
    NNDEPLOY_LOGE("TokenizerDecode::serialize failed\n");
    return status;
  }
  return base::kStatusCodeOk;
}

base::Status TokenizerEncodeMnn::deserialize(rapidjson::Value& json) {
  base::Status status = dag::Node::deserialize(json);
  if (status != base::kStatusCodeOk) {
    NNDEPLOY_LOGE("TokenizerDecode::deserialize failed\n");
    return status;
  }
  return base::kStatusCodeOk;
}

// TokenizerDecodeMnn 实现

TokenizerDecodeMnn::TokenizerDecodeMnn(const std::string& name)
    : TokenizerDecode(name) {
  key_ = "nndeploy::tokenizer::TokenizerDecodeMnn";
  desc_ =
      "A tokenizer decode node that  to "
      "decode token IDs into text. Supports HuggingFace and BPE "
      "tokenizers. "
      "Can decode single token IDs or batches of token IDs. Provides "
      "token-to-"
      "text conversion.";
  param_ = std::make_shared<TokenizerPraram>();
  // this->setInputTypeInfo<TokenizerIds>();
  // this->setOutputTypeInfo<TokenizerText>();
}
TokenizerDecodeMnn::TokenizerDecodeMnn(const std::string& name,
                                       std::vector<dag::Edge*> inputs,
                                       std::vector<dag::Edge*> outputs)
    : TokenizerDecode(name, inputs, outputs) {
  key_ = "nndeploy::tokenizer::TokenizerDecodeMnn";
  desc_ =
      "A tokenizer decode node that  to "
      "decode token IDs into text. Supports HuggingFace and BPE "
      "tokenizers. "
      "Can decode single token IDs or batches of token IDs. Provides "
      "token-to-"
      "text conversion.";
  param_ = std::make_shared<TokenizerPraram>();
  // this->setInputTypeInfo<TokenizerIds>();
  // this->setOutputTypeInfo<TokenizerText>();
}
TokenizerDecodeMnn::~TokenizerDecodeMnn() {
  // TODO: 清理资源
}

base::Status TokenizerDecodeMnn::init() {
  // TODO: 初始化MNN tokenizer
  return base::kStatusCodeOk;
}

base::Status TokenizerDecodeMnn::deinit() {
  // TODO: 反初始化MNN tokenizer
  return base::kStatusCodeOk;
}

base::Status TokenizerDecodeMnn::run() {
  // TODO: 实现运行逻辑
  return base::kStatusCodeOk;
}

std::string TokenizerDecodeMnn::decode(const std::vector<int32_t>& ids) {
  // TODO: 实现token ID解码
  return std::string();
}

std::vector<std::string> TokenizerDecodeMnn::decodeBatch(
    const std::vector<std::vector<int32_t>>& ids) {
  // TODO: 实现批量token ID解码
  return std::vector<std::string>();
}

size_t TokenizerDecodeMnn::getVocabSize() {
  // TODO: 返回词汇表大小
  return 0;
}

std::string TokenizerDecodeMnn::idToToken(int32_t token_id) {
  // TODO: 将ID转换为token
  return std::string();
}

base::Status TokenizerDecodeMnn::serialize(
    rapidjson::Value& json, rapidjson::Document::AllocatorType& allocator) {
  base::Status status = dag::Node::serialize(json, allocator);
  if (status != base::kStatusCodeOk) {
    NNDEPLOY_LOGE("TokenizerDecode::serialize failed\n");
    return status;
  }
  return base::kStatusCodeOk;
}
base::Status TokenizerDecodeMnn::deserialize(rapidjson::Value& json) {
  base::Status status = dag::Node::deserialize(json);
  if (status != base::kStatusCodeOk) {
    NNDEPLOY_LOGE("TokenizerDecode::deserialize failed\n");
    return status;
  }
  return base::kStatusCodeOk;
}

// REGISTER_NODE("nndeploy::tokenizer::TokenizerEncodeMnn", TokenizerEncodeMnn);
// REGISTER_NODE("nndeploy::tokenizer::TokenizerDecodeMnn", TokenizerDecodeMnn);

}  // namespace tokenizer
}  // namespace nndeploy
