
//
//  tokenizer.hpp
//
//  Created by MNN on 2023/09/25.
//  ZhaodeWang
//

/**
 * @ZhaodeWang
 * 基于mnn/transformers/llm/engine/src/tokenizer.hpp实现，有如下方案
 * 方案1：
 *    直接把mnn的实现代码抽出来，封装在两个类中
 * 方案2：
 *    在另外的文件种实现mnn/transformers/llm/engine/src/tokenizer.hpp的功能，这两个门面类中分别调用函数
 *
 * 讨论：
 * 1. 是否可以共用 TokenizerPraram 类
 * 2. TokenizerEncodeMnn和TokenizerDecodeMnn共用一份词表文件，如何优化？
 */

#ifndef _NNDEPLOY_TOKENIZER_TOKENIZER_MNN_TOKENIZER_MNN_H_
#define _NNDEPLOY_TOKENIZER_TOKENIZER_MNN_TOKENIZER_MNN_H_

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
 * @brief TokenizerEncodeMnn
 *
 */
class NNDEPLOY_CC_API TokenizerEncodeMnn
    : public nndeploy::tokenizer::TokenizerEncode {
 public:
  TokenizerEncodeMnn(const std::string& name);
  TokenizerEncodeMnn(const std::string& name, std::vector<dag::Edge*> inputs,
                     std::vector<dag::Edge*> outputs);
  virtual ~TokenizerEncodeMnn();

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

  // @ZhaodeWang：加入需要前端展示的参数，需要些serialize和deserialize方法
  using dag::Node::serialize;
  virtual base::Status serialize(
      rapidjson::Value& json,
      rapidjson::Document::AllocatorType& allocator) override;
  using dag::Node::deserialize;
  virtual base::Status deserialize(rapidjson::Value& json) override;

 private:
  // @ZhaodeWang: 添加成员变量
};

/**
 * @brief TokenizerDecodeMnn
 *
 */
class NNDEPLOY_CC_API TokenizerDecodeMnn
    : public nndeploy::tokenizer::TokenizerDecode {
 public:
  TokenizerDecodeMnn(const std::string& name);
  TokenizerDecodeMnn(const std::string& name, std::vector<dag::Edge*> inputs,
                     std::vector<dag::Edge*> outputs);
  virtual ~TokenizerDecodeMnn();

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

  // @ZhaodeWang：有需要前端展示的参数，需要写serialize和deserialize方法
  using dag::Node::serialize;
  virtual base::Status serialize(
      rapidjson::Value& json,
      rapidjson::Document::AllocatorType& allocator) override;
  using dag::Node::deserialize;
  virtual base::Status deserialize(rapidjson::Value& json) override;

 private:
  // @ZhaodeWang: 添加成员变量
};

}  // namespace tokenizer
}  // namespace nndeploy

#endif
