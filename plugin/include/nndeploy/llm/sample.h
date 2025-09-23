//
//  sample.hpp
//
//  Created by MNN on 2023/09/25.
//  ZhaodeWang
//

/**
 * @ZhaodeWang
 * 基于
 *  1. mnn/transformers/llm/engine/src/sampler.hpp
 *  2. mnn/transformers/llm/engine/src/sampler.cpp
 * 实现，有如下方案
 * 方案1：
 *    直接把mnn的实现代码抽出来，封装在两个类中
 * 方案2：
 *    在另外的文件种实现mnn代码的功能，这两个作为门面类，调用函数
 *
 */

#ifndef _NNDEPLOY_LLM_SAMPLE_H_
#define _NNDEPLOY_LLM_SAMPLE_H_

#include "nndeploy/base/any.h"
#include "nndeploy/base/common.h"
#include "nndeploy/base/glic_stl_include.h"
#include "nndeploy/base/log.h"
#include "nndeploy/base/macro.h"
#include "nndeploy/base/object.h"
#include "nndeploy/base/opencv_include.h"
#include "nndeploy/base/param.h"
#include "nndeploy/base/status.h"
#include "nndeploy/base/string.h"
#include "nndeploy/dag/composite_node.h"
#include "nndeploy/dag/edge.h"
#include "nndeploy/dag/graph.h"
#include "nndeploy/dag/loop.h"
#include "nndeploy/dag/node.h"
#include "nndeploy/device/buffer.h"
#include "nndeploy/device/device.h"
#include "nndeploy/device/memory_pool.h"
#include "nndeploy/device/tensor.h"
#include "nndeploy/tokenizer/tokenizer.h"

namespace nndeploy {
namespace llm {

/**
 * @brief SampleParam - Sample节点的参数配置
 */
class NNDEPLOY_CC_API SampleParam : public base::Param {
 public:
  SampleParam() = default;
  virtual ~SampleParam() = default;

  int max_new_tokens = 512;
  int max_all_tokens = 2048;
  std::string type = "temperature";
  std::string select_type = "temperature";
  float temperature = 0.8;
  int topK = 40;
  float topP = 0.9;
  float minP = 0.05;
  float tfsZ = 1.0;
  float typical = 0.95;
  // penalty
  float penalty = 1.05;
  int ngram = 8;
  float ngram_factor =
      1.02;  // panalize repeated ngram with a multiplied ngram_factor.
  float max_penalty = 10.0f;
  std::string sampler = "temperature";  // "greedy", "temperature".

  using base::Param::serialize;
  virtual base::Status serialize(
      rapidjson::Value& json,
      rapidjson::Document::AllocatorType& allocator) override;
  using base::Param::deserialize;
  virtual base::Status deserialize(rapidjson::Value& json) override;
};

/**
 * @brief Sample - 词嵌入节点
 *
 * 将输入的token ID序列转换为对应的嵌入向量表示
 *
 * 输入：
 * - inputs[0]: TokenizerIds - 包含token ID序列的数据结构
 *
 * 输出：
 * - outputs[0]: device::Tensor - 嵌入向量张量，形状为[batch_size, seq_len,
 * hidden_size]
 */
class NNDEPLOY_CC_API Sample : public dag::Node {
 public:
  Sample(const std::string& name, std::vector<dag::Edge*> inputs,
         std::vector<dag::Edge*> outputs);
  virtual ~Sample();

  virtual base::Status init();
  virtual base::Status deinit();

  virtual base::Status run();

  virtual base::Status defaultParam();

  using dag::Node::serialize;
  virtual base::Status serialize(
      rapidjson::Value& json,
      rapidjson::Document::AllocatorType& allocator) override;
  using dag::Node::deserialize;
  virtual base::Status deserialize(rapidjson::Value& json) override;
};

}  // namespace llm
}  // namespace nndeploy

#endif