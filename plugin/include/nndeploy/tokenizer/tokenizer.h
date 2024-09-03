
#ifndef _NNDEPLOY_MODEL_TOKENIZER_TOKENIZER_H_
#define _NNDEPLOY_MODEL_TOKENIZER_TOKENIZER_H_

#include "nndeploy/base/common.h"
#include "nndeploy/base/glic_stl_include.h"
#include "nndeploy/base/log.h"
#include "nndeploy/base/macro.h"
#include "nndeploy/base/object.h"
#include "nndeploy/base/param.h"
#include "nndeploy/base/status.h"
#include "nndeploy/base/string.h"
#include "nndeploy/base/type.h"
#include "nndeploy/base/value.h"
#include "nndeploy/dag/edge.h"
#include "nndeploy/dag/graph.h"
#include "nndeploy/dag/node.h"
#include "nndeploy/device/buffer.h"
#include "nndeploy/device/device.h"
#include "nndeploy/device/memory_pool.h"
#include "nndeploy/device/tensor.h"

namespace nndeploy {
namespace model {

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
    is_encode_ = tp.is_encode_;
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

  //  encode or decode
  bool is_encode_;
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
};

class NNDEPLOY_CC_API TokenizerText : public base::Param {
 public:
  std::vector<std::string> texts_;
};

class NNDEPLOY_CC_API TokenizerIds : public base::Param {
 public:
  std::vector<std::vector<int32_t>> ids_;
};

/**
 * @brief Tokenizer
 *
 */
class NNDEPLOY_CC_API Tokenizer : public dag::Node {
 public:
  Tokenizer(const std::string &name, dag::Edge *input, dag::Edge *output);

  virtual ~Tokenizer();
};

}  // namespace model
}  // namespace nndeploy

#endif /* _NNDEPLOY_MODEL_TOKENIZER_TOKENIZER_H_ */
