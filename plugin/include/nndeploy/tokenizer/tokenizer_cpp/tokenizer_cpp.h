
#ifndef _NNDEPLOY_TOKENIZER_TOKENIZER_TOKENIZER_CPP_TOKENIZER_CPP_H_
#define _NNDEPLOY_TOKENIZER_TOKENIZER_TOKENIZER_CPP_TOKENIZER_CPP_H_

#include <tokenizers_cpp.h>

#include "nndeploy/base/common.h"
#include "nndeploy/base/glic_stl_include.h"
#include "nndeploy/base/log.h"
#include "nndeploy/base/macro.h"
#include "nndeploy/base/object.h"
#include "nndeploy/base/param.h"
#include "nndeploy/base/status.h"
#include "nndeploy/base/string.h"
#include "nndeploy/base/type.h"
#include "nndeploy/base/any.h"
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
 * @brief Tokenizer
 *
 */
class NNDEPLOY_CC_API TokenizerCpp : public nndeploy::model::Tokenizer {
 public:
  TokenizerCpp(const std::string& name, dag::Edge* input, dag::Edge* output);
  virtual ~TokenizerCpp();

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

  /*!
   * \brief Convert the given token to its corresponding id if it exists. If
   * not, return -1.
   */
  int32_t tokenToId(const std::string& token);

 private:
  std::unique_ptr<tokenizers::Tokenizer> tokenizer_;
};

}  // namespace tokenizer
}  // namespace nndeploy

#endif /* _NNDEPLOY_TOKENIZER_TOKENIZER_TOKENIZER_H_ */
