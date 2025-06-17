
#ifndef _NNDEPLOY_TOKENIZER_TOKENIZER_TOKENIZER_H_
#define _NNDEPLOY_TOKENIZER_TOKENIZER_TOKENIZER_H_

#include "nndeploy/base/any.h"
#include "nndeploy/base/common.h"
#include "nndeploy/base/glic_stl_include.h"
#include "nndeploy/base/log.h"
#include "nndeploy/base/macro.h"
#include "nndeploy/base/object.h"
#include "nndeploy/base/param.h"
#include "nndeploy/base/status.h"
#include "nndeploy/base/string.h"
#include "nndeploy/base/type.h"
#include "nndeploy/dag/edge.h"
#include "nndeploy/dag/graph.h"
#include "nndeploy/dag/node.h"
#include "nndeploy/device/buffer.h"
#include "nndeploy/device/device.h"
#include "nndeploy/device/memory_pool.h"
#include "nndeploy/device/tensor.h"

namespace nndeploy {
namespace tokenizer {

//---------------------------------------------------
// Factory functions from byte-blobs
// These factory function takes in in-memory blobs
// so the library can be independent from filesystem
//---------------------------------------------------
enum TokenizerType : int {
  /*!
   * \brief Create HF tokenizer from a single in-memory json blob.
   *
   * \param json_blob The json blob.
   * \return The created tokenzier.
   */
  kTokenizerTypeHF = 0x0000,
  /*!
   * \brief Create BPE tokenizer
   *
   * \param vocab_blob The blob that contains vocabs.
   * \param merges_blob The blob that contains the merges.
   * \param added_tokens The added tokens.
   * \return The created tokenizer.
   */
  kTokenizerTypeBPE,
  /*!
   * \brief Create SentencePiece.
   *
   * \param model_blob The blob that contains vocabs.
   * \return The created tokenizer.
   */
  kTokenizerTypeSentencePiece,
  /*!
   * \brief Create RWKVWorldTokenizer.
   *
   * \param model_blob The blob that contains vocabs.
   * \return The created tokenizer.
   */
  kTokenizerTypeRWKVWorld,
  kTokenizerTypeNotSupport,
};

extern NNDEPLOY_CC_API std::string tokenizerTypeToString(TokenizerType type);
extern NNDEPLOY_CC_API TokenizerType stringToTokenizerType(const std::string &src);

class NNDEPLOY_CC_API TokenizerPraram : public base::Param {
 public:
  TokenizerPraram() : base::Param() {}
  virtual ~TokenizerPraram() {}

  PARAM_COPY(TokenizerPraram)
  PARAM_COPY_TO(TokenizerPraram)

  TokenizerPraram &operator=(const TokenizerPraram &tp) {
    if (this == &tp) {
      return *this;
    }
    is_path_ = tp.is_path_;
    tokenizer_type_ = tp.tokenizer_type_;
    json_blob_ = tp.json_blob_;
    model_blob_ = tp.model_blob_;
    vocab_blob_ = tp.vocab_blob_;
    merges_blob_ = tp.merges_blob_;
    added_tokens_ = tp.added_tokens_;
    max_length_ = tp.max_length_;

    return *this;
  }

  // is_path
  bool is_path_ = true;
  // The type of tokenizer
  TokenizerType tokenizer_type_;

  /*!
   * \brief Create HF tokenizer from a single in-memory json blob.
   *
   * \param json_blob The json blob.
   * \return The created tokenzier.
   */
  std::string json_blob_;
  /*!
   * \brief Create SentencePiece.
   *
   * \param model_blob The blob that contains vocabs.
   * \return The created tokenizer.
   */
  /*!
   * \brief Create RWKVWorldTokenizer.
   *
   * \param model_blob The blob that contains vocabs.
   * \return The created tokenizer.
   */
  std::string model_blob_;
  /*!
   * \brief Create BPE tokenizer
   *
   * \param vocab_blob The blob that contains vocabs.
   * \param merges_blob The blob that contains the merges.
   * \param added_tokens The added tokens.
   * \return The created tokenizer.
   */
  std::string vocab_blob_;
  std::string merges_blob_;
  std::string added_tokens_;

  int max_length_ = 77;

  using base::Param::serialize;
  virtual base::Status serialize(
      rapidjson::Value& json,
      rapidjson::Document::AllocatorType& allocator) override {
    json.AddMember("is_path_", is_path_, allocator);
    std::string tokenizer_type_str = tokenizerTypeToString(tokenizer_type_);
    json.AddMember("tokenizer_type_", rapidjson::Value(tokenizer_type_str.c_str(), allocator), allocator);
    json.AddMember("json_blob_", rapidjson::Value(json_blob_.c_str(), allocator), allocator);
    json.AddMember("model_blob_", rapidjson::Value(model_blob_.c_str(), allocator), allocator);
    json.AddMember("vocab_blob_", rapidjson::Value(vocab_blob_.c_str(), allocator), allocator);
    json.AddMember("merges_blob_", rapidjson::Value(merges_blob_.c_str(), allocator), allocator);
    json.AddMember("added_tokens_", rapidjson::Value(added_tokens_.c_str(), allocator), allocator);
    json.AddMember("max_length_", max_length_, allocator);
    return base::kStatusCodeOk;

  }

  using base::Param::deserialize;
  virtual base::Status deserialize(rapidjson::Value& json) override {
    if (json.HasMember("is_path_") && json["is_path_"].IsBool()) {
      is_path_ = json["is_path_"].GetBool();
    }
    if (json.HasMember("tokenizer_type_") && json["tokenizer_type_"].IsString()) {
      tokenizer_type_ = stringToTokenizerType(json["tokenizer_type_"].GetString());
    }
    if (json.HasMember("json_blob_") && json["json_blob_"].IsString()) {
      json_blob_ = json["json_blob_"].GetString();
    }
    if (json.HasMember("model_blob_") && json["model_blob_"].IsString()) {
      model_blob_ = json["model_blob_"].GetString();
    }
    if (json.HasMember("vocab_blob_") && json["vocab_blob_"].IsString()) {
      vocab_blob_ = json["vocab_blob_"].GetString();
    }
    if (json.HasMember("merges_blob_") && json["merges_blob_"].IsString()) {
      merges_blob_ = json["merges_blob_"].GetString();
    }
    if (json.HasMember("added_tokens_") && json["added_tokens_"].IsString()) {
      added_tokens_ = json["added_tokens_"].GetString();
    }
    if (json.HasMember("max_length_") && json["max_length_"].IsInt()) {
      max_length_ = json["max_length_"].GetInt();
    }
    return base::kStatusCodeOk;
  }
};

class NNDEPLOY_CC_API TokenizerText : public base::Param {
 public:
  std::vector<std::string> texts_;

  using base::Param::serialize; 
  virtual base::Status serialize(
      rapidjson::Value& json,
      rapidjson::Document::AllocatorType& allocator) override {
    rapidjson::Value texts_json(rapidjson::kArrayType);
    for (const auto& text : texts_) {
      texts_json.PushBack(rapidjson::Value(text.c_str(), allocator), allocator);
    }
    json.AddMember("texts_", texts_json, allocator);
    return base::kStatusCodeOk;
  }

  using base::Param::deserialize;
  virtual base::Status deserialize(rapidjson::Value& json) override {
    if (json.HasMember("texts_") && json["texts_"].IsArray()) {
      texts_.clear();
      for (const auto& text : json["texts_"].GetArray()) {
        texts_.push_back(text.GetString());
      }
    }
    return base::kStatusCodeOk;
  }
};

class NNDEPLOY_CC_API TokenizerIds : public base::Param {
 public:
  std::vector<std::vector<int32_t>> ids_;
};

/**
 * @brief TokenizerEncode
 *
 */
class NNDEPLOY_CC_API TokenizerEncode : public dag::Node {
 public:
  TokenizerEncode(const std::string &name) : dag::Node(name) {
    this->setInputTypeInfo<TokenizerText>();
    this->setOutputTypeInfo<TokenizerIds>();
  }
  TokenizerEncode(const std::string &name, std::vector<dag::Edge *> inputs,
                  std::vector<dag::Edge *> outputs)
      : dag::Node(name, inputs, outputs) {
    this->setInputTypeInfo<TokenizerText>();
    this->setOutputTypeInfo<TokenizerIds>();
  }

  virtual ~TokenizerEncode();

  virtual base::Status run() = 0;
};

/**
 * @brief TokenizerDecode
 *
 */
class NNDEPLOY_CC_API TokenizerDecode : public dag::Node {
 public:
  TokenizerDecode(const std::string &name) : dag::Node(name) {
    this->setInputTypeInfo<TokenizerIds>();
    this->setOutputTypeInfo<TokenizerText>();
  }
  TokenizerDecode(const std::string &name, std::vector<dag::Edge *> inputs,
                  std::vector<dag::Edge *> outputs)
      : dag::Node(name, inputs, outputs) {
    this->setInputTypeInfo<TokenizerIds>();
    this->setOutputTypeInfo<TokenizerText>();
  }

  virtual ~TokenizerDecode();

  virtual base::Status run() = 0;
};

}  // namespace tokenizer
}  // namespace nndeploy

#endif /* _NNDEPLOY_TOKENIZER_TOKENIZER_TOKENIZER_H_ */
