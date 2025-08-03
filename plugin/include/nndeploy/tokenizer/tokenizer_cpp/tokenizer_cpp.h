
#ifndef _NNDEPLOY_TOKENIZER_TOKENIZER_TOKENIZER_CPP_TOKENIZER_CPP_H_
#define _NNDEPLOY_TOKENIZER_TOKENIZER_TOKENIZER_CPP_TOKENIZER_CPP_H_

#include <tokenizers_cpp.h>

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
#include "nndeploy/tokenizer/tokenizer.h"

namespace nndeploy {
namespace tokenizer {

/**
 * @brief TokenizerEncodeCpp
 *
 */
class NNDEPLOY_CC_API TokenizerEncodeCpp
    : public nndeploy::tokenizer::TokenizerEncode {
 public:
  TokenizerEncodeCpp(const std::string& name) : TokenizerEncode(name) {
    key_ = "nndeploy::tokenizer::TokenizerEncodeCpp";
    desc_ =
        "A tokenizer encode node that uses the C++ tokenizers library to "
        "encode text into token IDs. Supports HuggingFace and BPE tokenizers. "
        "Can encode single strings or batches of text. Provides vocabulary "
        "lookup and token-to-ID conversion.";
    param_ = std::make_shared<TokenizerPraram>();
    // this->setInputTypeInfo<TokenizerText>();
    // this->setOutputTypeInfo<TokenizerIds>();
  }
  TokenizerEncodeCpp(const std::string& name, std::vector<dag::Edge*> inputs,
                     std::vector<dag::Edge*> outputs)
      : TokenizerEncode(name, inputs, outputs) {
    key_ = "nndeploy::tokenizer::TokenizerEncodeCpp";
    desc_ =
        "A tokenizer encode node that uses the C++ tokenizers library to "
        "encode text into token IDs. Supports HuggingFace and BPE tokenizers. "
        "Can encode single strings or batches of text. Provides vocabulary "
        "lookup and token-to-ID conversion.";
    param_ = std::make_shared<TokenizerPraram>();
    // this->setInputTypeInfo<TokenizerText>();
    // this->setOutputTypeInfo<TokenizerIds>();
  }
  virtual ~TokenizerEncodeCpp();

  virtual base::Status init();
  virtual base::Status deinit();

  virtual base::Status run();

  /*!
   * \brief encode text into ids.
   * \param text The input text.
   * \returns The encoded token ids.
   */
  std::vector<int32_t> encode(const std::string& text);

  /*!
   * \brief encode a batch of texts into ids.
   * \param texts The input texts.
   * \returns The encoded token ids.
   */
  std::vector<std::vector<int32_t>> encodeBatch(
      const std::vector<std::string>& texts);

  /*!
   * \brief Returns the vocabulary size. Special tokens are considered.
   */
  size_t getVocabSize();

  /*!
   * \brief Convert the given token to its corresponding id if it exists. If
   * not, return -1.
   */
  int32_t tokenToId(const std::string& token);

 private:
  std::unique_ptr<tokenizers::Tokenizer> tokenizer_;
};

/**
 * @brief TokenizerDecodeCpp
 *
 */
class NNDEPLOY_CC_API TokenizerDecodeCpp
    : public nndeploy::tokenizer::TokenizerDecode {
 public:
  TokenizerDecodeCpp(const std::string& name) : TokenizerDecode(name) {
    key_ = "nndeploy::tokenizer::TokenizerDecodeCpp";
    desc_ =
        "A tokenizer decode node that uses the C++ tokenizers library to "
        "decode token IDs into text. Supports HuggingFace and BPE tokenizers. "
        "Can decode single token IDs or batches of token IDs. Provides token-to-"
        "text conversion.";
    param_ = std::make_shared<TokenizerPraram>();
    // this->setInputTypeInfo<TokenizerIds>();
    // this->setOutputTypeInfo<TokenizerText>();
  }
  TokenizerDecodeCpp(const std::string& name, std::vector<dag::Edge*> inputs,
                     std::vector<dag::Edge*> outputs)
      : TokenizerDecode(name, inputs, outputs) {
    key_ = "nndeploy::tokenizer::TokenizerDecodeCpp";
    desc_ =
        "A tokenizer decode node that uses the C++ tokenizers library to "
        "decode token IDs into text. Supports HuggingFace and BPE tokenizers. "
        "Can decode single token IDs or batches of token IDs. Provides token-to-"
        "text conversion.";
    param_ = std::make_shared<TokenizerPraram>();
    // this->setInputTypeInfo<TokenizerIds>();
    // this->setOutputTypeInfo<TokenizerText>();
  }
  virtual ~TokenizerDecodeCpp();

  virtual base::Status init();
  virtual base::Status deinit();

  virtual base::Status run();

  /*!
   * \brief decode token ids into text.
   * \param text The token ids.
   * \returns The decoded text.
   */
  std::string decode(const std::vector<int32_t>& ids);

  /*!
   * \brief decode a batch of token ids into text.
   * \param text The token ids.
   * \returns The decoded text.
   */
  std::vector<std::string> decodeBatch(
      const std::vector<std::vector<int32_t>>& ids);

  /*!
   * \brief Returns the vocabulary size. Special tokens are considered.
   */
  size_t getVocabSize();

  /*!
   * \brief Convert the given id to its corresponding token if it exists. If
   * not, return an empty string.
   */
  std::string idToToken(int32_t token_id);

 private:
  std::unique_ptr<tokenizers::Tokenizer> tokenizer_;
};

}  // namespace tokenizer
}  // namespace nndeploy

#endif /* _NNDEPLOY_TOKENIZER_TOKENIZER_TOKENIZER_H_ */
