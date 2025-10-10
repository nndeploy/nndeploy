// 封装MNN LLM推理引擎

/**
 * @brief MNN LLM推理引擎封装
 *
 * 基于MNN框架实现的大语言模型推理引擎，提供完整的LLM推理能力
 * 参考MNN transformers/llm/engine实现
 *
 *
 * 支持特性：
 * - 量化推理（INT4/INT8）
 * - 动态批处理
 * - KV缓存优化
 * - 流式生成
 */

#ifndef _NNDEPLOY_LLM_MNN_MNN_LLM_INFER_H_
#define _NNDEPLOY_LLM_MNN_MNN_LLM_INFER_H_

#include <MNN/AutoTime.hpp>
#include <MNN/expr/ExecutorScope.hpp>

#include "MNN/llm/llm.hpp"
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
#include "nndeploy/inference/mnn/mnn_convert.h"
#include "nndeploy/llm/abstract_llm_infer.h"

namespace nndeploy {
namespace llm {

/**
 * @brief MnnLlmInfer - MNN LLM推理节点
 *
 * 基于MNN框架的大语言模型推理节点，继承自AbstractLlmInfer
 * 提供完整的文本生成能力，支持多种采样策略和优化特性
 *
 * 输入：
 * - inputs[0]: tokenizer::TokenizerIds - 输入token序列
 *
 * 输出：
 * - outputs[0]: device::Tensor - 输出logits张量
 */
class NNDEPLOY_CC_API MnnLlmInfer : public AbstractLlmInfer {
 public:
  MnnLlmInfer(const std::string& name) : AbstractLlmInfer(name) {}
  MnnLlmInfer(const std::string& name, std::vector<dag::Edge*> inputs,
              std::vector<dag::Edge*> outputs)
      : AbstractLlmInfer(name, inputs, outputs) {}
  virtual ~MnnLlmInfer() {}

  virtual base::Status init() override {
    std::string share_key = getShareKey();
    auto infer =
        this->getResourceWithoutState<std::shared_ptr<MNN::Transformer::Llm>>(
            share_key);
    if (infer == nullptr) {
      MNN::BackendConfig backendConfig;
      executor_ = MNN::Express::Executor::newExecutor(MNN_FORWARD_CPU,
                                                      backendConfig, 1);
      MNN::Express::ExecutorScope s(executor_);

      mnn_llm_ = std::shared_ptr<MNN::Transformer::Llm>(
          MNN::Transformer::Llm::createLLM(config_path_[0]));
      std::cout << "config path is " << config_path_[0] << std::endl;
      mnn_llm_->set_config("{\"tmp_path\":\"tmp\"}");
      {
        bool res = mnn_llm_->load();
        if (!res) {
          NNDEPLOY_LOGE("LLM init error\n");
          return base::kStatusCodeErrorInvalidParam;
        }
      }
      if (true) {
        NNDEPLOY_LOGI("Prepare for tuning opt Begin\n");
        mnn_llm_->tuning(MNN::Transformer::OP_ENCODER_NUMBER,
                         {1, 5, 10, 20, 30, 50, 100});
        NNDEPLOY_LOGI("Prepare for tuning opt End\n");
      }
      this->addResourceWithoutState(share_key, mnn_llm_);
    } else {
      mnn_llm_ = infer;
    }
    return base::kStatusCodeOk;
  }
  virtual base::Status deinit() override { return base::kStatusCodeOk; }
  virtual base::Status run() override {
    MNN::Express::ExecutorScope s(executor_);
    if (is_prefill_) {
      return prefill();
    } else {
      return decode();
    }
  }

  virtual base::Status prefill() {
    // 全局的history_token
    tokenizer::TokenizerIds* ids =
        (tokenizer::TokenizerIds*)inputs_[0]->getParam(this);
    std::vector<int32_t>* history_tokens =
        new std::vector<int32_t>(ids->ids_[0]);
    dag::Edge* history_tokens_edge =
        this->createResourceWithState("history_tokens");
    history_tokens_edge->set<std::vector<int32_t>>(history_tokens, false);

    std::vector<int> input_ids = ids->ids_[0];

    output_logits_ = mnn_llm_->forward(input_ids, true);

    device::Tensor* output_logits =
        inference::convertToTensor(output_logits_, outputs_[0]->getName(),
                                   device::getDefaultHostDevice(), false);
    outputs_[0]->set(output_logits, false);

    return base::kStatusCodeOk;
  }
  virtual base::Status decode() {  // 执行embedding节点和infer节点
    tokenizer::TokenizerIds* ids = nullptr;
    if (inputs_.size() == 1 || inputs_[1]->empty()) {
      ids = (tokenizer::TokenizerIds*)inputs_[0]->getParam(this);
    } else {
      ids = (tokenizer::TokenizerIds*)inputs_[1]->getParam(this);
    }
    dag::Edge* history_tokens_edge =
        this->getResourceWithState("history_tokens");
    std::vector<int32_t>* history_tokens = nullptr;
    if (history_tokens_edge != nullptr) {
      history_tokens = history_tokens_edge->get<std::vector<int32_t>>(this);
      history_tokens->push_back(ids->ids_[0].back());
    }

    std::vector<int> input_ids = {ids->ids_[0].back()};

    output_logits_ = mnn_llm_->forward(input_ids, false);

    device::Tensor* output_logits =
        inference::convertToTensor(output_logits_, outputs_[0]->getName(),
                                   device::getDefaultHostDevice(), false);
    outputs_[0]->set(output_logits, false);

    return base::kStatusCodeOk;
  }

 private:
  // MNN相关成员变量
  std::shared_ptr<MNN::Transformer::Llm> mnn_llm_;  // MNN LLM实例
  MNN::Express::VARP output_logits_;
  std::shared_ptr<MNN::Express::Executor> executor_;
};

}  // namespace llm
}  // namespace nndeploy

#endif  // _NNDEPLOY_LLM_MNN_MNN_LLM_INFER_H_