
#include "nndeploy/model/tokenizer/tokenizer_cpp/tokenizer_cpp.h"

namespace nndeploy {
namespace model {

TokenizerCpp::TokenizerCpp(const std::string& name, dag::Edge* input,
                           dag::Edge* output)
    : Tokenizer(name, input, output) {
  param_ = std::make_shared<TokenizerPraram>();
}
TokenizerCpp::~TokenizerCpp() {}

base::Status TokenizerCpp::init() {
  base::Status status = base::kStatusCodeOk;
  NNDEPLOY_LOGE("test tokenizer_cpp\n");

  // param_
  TokenizerPraram* tokenizer_param = (TokenizerPraram*)(param_.get());
  if (tokenizer_param->tokenizer_type_ == TokenizerType::kTokenizerTypeHF) {
    tokenizer_ =
        tokenizers::Tokenizer::FromBlobJSON(tokenizer_param->json_blob_);
  } else if (tokenizer_param->tokenizer_type_ ==
             TokenizerType::kTokenizerTypeBPE) {
    tokenizer_ = tokenizers::Tokenizer::FromBlobByteLevelBPE(
        tokenizer_param->vocab_blob_, tokenizer_param->merges_blob_,
        tokenizer_param->added_tokens_);
  } else if (tokenizer_param->tokenizer_type_ ==
             TokenizerType::kTokenizerTypeSentencePiece) {
    NNDEPLOY_LOGE("test tokenizer_cpp\n");
    NNDEPLOY_LOGE("%s\n", tokenizer_param->model_blob_.c_str());
    tokenizer_ = tokenizers::Tokenizer::FromBlobSentencePiece(
        tokenizer_param->model_blob_);
    NNDEPLOY_LOGE("test tokenizer_cpp\n");
  } else if (tokenizer_param->tokenizer_type_ ==
             TokenizerType::kTokenizerTypeRWKVWorld) {
    tokenizer_ = tokenizers::Tokenizer::FromBlobSentencePiece(
        tokenizer_param->model_blob_);
  } else {
    status = base::kStatusCodeErrorInvalidParam;
    NNDEPLOY_RETURN_ON_NEQ(status, base::kStatusCodeOk,
                           "Invalid tokenizer type!");
  }

  return status;
}

base::Status TokenizerCpp::deinit() {
  base::Status status = base::kStatusCodeOk;
  return status;
}

base::Status TokenizerCpp::run() {
  base::Status status = base::kStatusCodeOk;

  // param_
  TokenizerPraram* tokenizer_param = (TokenizerPraram*)(param_.get());
  bool is_encode = tokenizer_param->is_encode_;

  // run
  if (is_encode) {
    TokenizerText* text_param = (TokenizerText*)(inputs_[0]->getParam(this));
    int index = inputs_[0]->getIndex(this);
    TokenizerIds* ids_param = new TokenizerIds();
    ids_param->ids_ = encodeBatch(text_param->texts_);
    outputs_[0]->set(ids_param, index, false);
  } else {
    TokenizerIds* ids_param = (TokenizerIds*)(inputs_[0]->getParam(this));
    int index = inputs_[0]->getIndex(this);
    TokenizerText* text_param = new TokenizerText();
    text_param->texts_ = decodeBatch(ids_param->ids_);
    outputs_[0]->set(text_param, index, false);
  }

  return status;
}

/*!
 * \brief encode text into ids.
 * \param text The input text.
 * \returns The encoded token ids.
 */
std::vector<int32_t> TokenizerCpp::encode(const std::string& text) {
  return tokenizer_->Encode(text);
}

/*!
 * \brief encode a batch of texts into ids.
 * \param texts The input texts.
 * \returns The encoded token ids.
 */
std::vector<std::vector<int32_t>> TokenizerCpp::encodeBatch(
    const std::vector<std::string>& texts) {
  // Fall back when the derived class does not implement this function.
  std::vector<std::vector<int32_t>> ret;
  ret.reserve(texts.size());
  for (const auto& text : texts) {
    ret.push_back(encode(text));
  }
  return ret;
}

/*!
 * \brief decode token ids into text.
 * \param text The token ids.
 * \returns The decoded text.
 */
std::string TokenizerCpp::decode(const std::vector<int32_t>& ids) {
  return tokenizer_->Decode(ids);
}

/*!
 * \brief decode a batch of token ids into text.
 * \param text The token ids.
 * \returns The decoded text.
 */
std::vector<std::string> TokenizerCpp::decodeBatch(
    const std::vector<std::vector<int32_t>>& ids) {
  // Fall back when the derived class does not implement this function.
  std::vector<std::string> ret;
  ret.reserve(ids.size());
  for (const auto& id : ids) {
    ret.push_back(decode(id));
  }
  return ret;
}

/*!
 * \brief Returns the vocabulary size. Special tokens are considered.
 */
size_t TokenizerCpp::getVocabSize() { return tokenizer_->GetVocabSize(); }

/*!
 * \brief Convert the given id to its corresponding token if it exists. If
 * not, return an empty string.
 */
std::string TokenizerCpp::idToToken(int32_t token_id) {
  return tokenizer_->IdToToken(token_id);
}

/*!
 * \brief Convert the given token to its corresponding id if it exists. If
 * not, return -1.
 */
int32_t TokenizerCpp::tokenToId(const std::string& token) {
  return tokenizer_->TokenToId(token);
}

}  // namespace model
}  // namespace nndeploy
