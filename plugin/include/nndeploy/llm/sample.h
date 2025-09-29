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
 * @wangzhaode
 */
class NNDEPLOY_CC_API SampleParam : public base::Param {
 public:
  SampleParam() = default;
  virtual ~SampleParam() = default;

  //  * 1. greedy - 贪婪采样，选择概率最高的token
  //  * 2. temperature - 温度采样，通过温度参数控制随机性
  //  * 3. topK - Top-K采样，从概率最高的K个token中采样
  //  * 4. topP - Top-P采样（核采样），从累积概率达到P的token集合中采样
  //  * 5. minP - Min-P采样，过滤掉概率低于阈值的token
  //  * 6. tfs - Tail Free Sampling，基于二阶导数的采样方法
  //  * 7. typical - Typical采样，基于信息论的采样方法
  //  * 8. penalty - 重复惩罚采样，对重复token进行惩罚
  //  * 9. ngram - N-gram重复惩罚，对重复的n-gram序列进行惩罚
  std::string sampler =
      "temperature";  // "greedy", "temperature", "topK", "topP", "minP", "tfs",
                      // "typical", "penalty", "ngram".

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
  std::vector<std::string> mixed_samplers = {"topK", "tfs",  "typical",
                                             "topP", "minP", "temperature"};

  using base::Param::serialize;
  virtual base::Status serialize(
      rapidjson::Value& json,
      rapidjson::Document::AllocatorType& allocator) override;
  using base::Param::deserialize;
  virtual base::Status deserialize(rapidjson::Value& json) override;
};

/**
 * @brief Sample - 文本生成采样节点
 *
 * 基于模型输出的logits进行采样，生成下一个token。支持多种采样算法：
 * 1. greedy - 贪婪采样，选择概率最高的token
 * 2. temperature - 温度采样，通过温度参数控制随机性
 * 3. top_k - Top-K采样，从概率最高的K个token中采样
 * 4. top_p - Top-P采样（核采样），从累积概率达到P的token集合中采样
 * 5. min_p - Min-P采样，过滤掉概率低于阈值的token
 * 6. tfs - Tail Free Sampling，基于二阶导数的采样方法
 * 7. typical - Typical采样，基于信息论的采样方法
 * 8. penalty - 重复惩罚采样，对重复token进行惩罚
 * 9. ngram - N-gram重复惩罚，对重复的n-gram序列进行惩罚
 *
 * 输入：
 * - inputs[0]: device::Tensor - 模型输出的logits张量，形状为[batch_size,
 * vocab_size]
 *
 * 输出：
 * - outputs[0]: int - 采样得到的下一个token ID
 */
class NNDEPLOY_CC_API Sampler : public dag::Node {
 public:
  Sampler(const std::string& name, std::vector<dag::Edge*> inputs,
          std::vector<dag::Edge*> outputs);
  virtual ~Sampler();

  virtual base::Status run();

  int sample(device::Tensor* logits);

  struct SubsetLogits penalty(struct SubsetLogits superset);
  struct SubsetLogits topK(struct SubsetLogits superset);
  struct SubsetLogits topP(struct SubsetLogits superset);
  struct SubsetLogits minP(struct SubsetLogits superset);
  struct SubsetLogits tfs(struct SubsetLogits superset);
  struct SubsetLogits typical(struct SubsetLogits superset);
  struct SubsetLogits mixed(struct SubsetLogits subset);
  struct SubsetLogits subsetSampler(std::string sampler_type,
                                    struct SubsetLogits subset);
  int handleSelect(struct SubsetLogits subset);
};

}  // namespace llm
}  // namespace nndeploy

#endif