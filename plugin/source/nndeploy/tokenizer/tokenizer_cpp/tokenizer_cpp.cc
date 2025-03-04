
#include "nndeploy/tokenizer/tokenizer_cpp/tokenizer_cpp.h"

#include "nndeploy/base/file.h"

namespace nndeploy {
namespace tokenizer {

TokenizerCpp::~TokenizerCpp() {}

std::string LoadBytesFromFile(const std::string& path) {
  std::ifstream fs(path, std::ios::in | std::ios::binary);
  if (fs.fail()) {
    std::cerr << "Cannot open " << path << std::endl;
    exit(1);
  }
  std::string data;
  fs.seekg(0, std::ios::end);
  size_t size = static_cast<size_t>(fs.tellg());
  fs.seekg(0, std::ios::beg);
  data.resize(size);
  fs.read(data.data(), size);
  return data;
}

base::Status TokenizerCpp::init() {
  base::Status status = base::kStatusCodeOk;

  // param_
  TokenizerPraram* tokenizer_param = (TokenizerPraram*)(param_.get());

  if (tokenizer_param->tokenizer_type_ == TokenizerType::kTokenizerTypeHF) {
    if (tokenizer_param->json_blob_.empty()) {
      NNDEPLOY_LOGE("json_blob_ is empty\n");
      return base::kStatusCodeErrorInvalidParam;
    }
    // Read blob from file.
    std::string blob;
    if (tokenizer_param->is_path_) {
      blob = base::openFile(tokenizer_param->json_blob_);
    } else {
      blob = tokenizer_param->json_blob_;
    }

    blob = LoadBytesFromFile(blob);
    tokenizer_ = tokenizers::Tokenizer::FromBlobJSON(blob);
  } else if (tokenizer_param->tokenizer_type_ ==
             TokenizerType::kTokenizerTypeBPE) {
    if (tokenizer_param->vocab_blob_.empty() ||
        tokenizer_param->merges_blob_.empty()) {
      NNDEPLOY_LOGE("vocab_blob_ or  merges_blob_ is empty\n");
      return base::kStatusCodeErrorInvalidParam;
    }
    // Read blob from file.
    std::string vocab_blob;
    std::string merges_blob;
    std::string added_tokens;
    if (tokenizer_param->is_path_) {
      vocab_blob = base::openFile(tokenizer_param->vocab_blob_);
      merges_blob = base::openFile(tokenizer_param->merges_blob_);
      added_tokens = base::openFile(tokenizer_param->added_tokens_);
    } else {
      vocab_blob = tokenizer_param->vocab_blob_;
      merges_blob = tokenizer_param->merges_blob_;
      added_tokens = tokenizer_param->added_tokens_;
    }
    tokenizer_ = tokenizers::Tokenizer::FromBlobByteLevelBPE(
        vocab_blob, merges_blob, added_tokens);
  } else if (tokenizer_param->tokenizer_type_ ==
             TokenizerType::kTokenizerTypeSentencePiece) {
    if (tokenizer_param->model_blob_.empty()) {
      NNDEPLOY_LOGE("model_blob_ is empty\n");
      return base::kStatusCodeErrorInvalidParam;
    }
    // Read blob from file.
    std::string blob;
    if (tokenizer_param->is_path_) {
      blob = base::openFile(tokenizer_param->model_blob_);
    } else {
      blob = tokenizer_param->model_blob_;
    }
    tokenizer_ = tokenizers::Tokenizer::FromBlobSentencePiece(blob);
  } else if (tokenizer_param->tokenizer_type_ ==
             TokenizerType::kTokenizerTypeRWKVWorld) {
    if (tokenizer_param->model_blob_.empty()) {
      NNDEPLOY_LOGE("model_blob_ is empty\n");
      return base::kStatusCodeErrorInvalidParam;
    }
    // Read blob from file.
    std::string blob;
    if (tokenizer_param->is_path_) {
      // blob = base::openFile(tokenizer_param->model_blob_);
      blob = tokenizer_param->model_blob_;
    } else {
      NNDEPLOY_LOGE("model_blob_ is in-memory\n");
      return base::kStatusCodeErrorInvalidParam;
    }
    tokenizer_ = tokenizers::Tokenizer::FromBlobRWKVWorld(blob);
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

}  // namespace tokenizer
}  // namespace nndeploy
