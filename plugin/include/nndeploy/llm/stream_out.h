
#ifndef _NNDEPLOY_LLM_STREAM_OUT_H_
#define _NNDEPLOY_LLM_STREAM_OUT_H_

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
 * @brief Stream - 大语言模型流式输出节点
 *
 * 该节点负责处理LLM的流式输出，支持实时token生成和输出
 *
 * 输入：
 * - inputs[0]: tokenizer::TokenizerIds - 生成的token ID
 *
 * 输出：
 * - outputs[0]: tokenizer::TokenizerText - 解码后的文本片段
 * - outputs[1]: std::string - 流式输出文本
 */
class NNDEPLOY_CC_API StreamOut : public dag::Node {
 public:
  StreamOut(const std::string& name, std::vector<dag::Edge*> inputs,
            std::vector<dag::Edge*> outputs)
      : dag::Node(name, inputs, outputs) {
    key_ = "nndeploy::llm::StreamOut";
    desc_ = "StreamOut: Stream output node";
    this->setDynamicOutput(true);
    this->setInputTypeInfo<tokenizer::TokenizerText>("input_text");
    this->setOutputTypeInfo<tokenizer::TokenizerText>("output_text");
    // this->setOutputTypeInfo<std::string>("stream_output");
  }

  virtual ~StreamOut() {}

  virtual base::Status init() override {
    is_first_ = true;
    return base::kStatusCodeOk;
  };

  bool isStopTexts() {
    // TODO，大语言模型的输出字符是什么？
    std::string text = *stream_output_;
    std::string last_text = "";
    size_t last_space = text.find_last_of(' ');
    if (last_space != std::string::npos) {
      last_text = text.substr(last_space + 1);
    } else {
      last_text = text;
    }
    return std::find(stop_texts_.begin(), stop_texts_.end(), last_text) !=
           stop_texts_.end();
  }

  virtual base::Status run() override {
    tokenizer::TokenizerText* input_text =
        dynamic_cast<tokenizer::TokenizerText*>(inputs_[0]->getParam(this));
    if (is_first_) {
      if (outputs_.size() > 1) {
        stream_output_ = new std::string();
      }
      if (enable_stream_) {
        NNDEPLOY_PRINTF("A: ");
      }
      output_text_ = new tokenizer::TokenizerText();
      output_text_->texts_.resize(input_text->texts_.size());
      is_first_ = false;
    }

    if (outputs_.size() > 1) {
      *stream_output_ = input_text->texts_[0];
    }

    if (isStopTexts()) {
      outputs_[0]->set(output_text_, true);
      if (enable_stream_) {
        NNDEPLOY_PRINTF("\n");
      }
    } else {
      output_text_->texts_[0] += (*stream_output_);
      if (enable_stream_) {
        NNDEPLOY_PRINTF("%s", input_text->texts_[0].c_str());
      }
    }

    if (outputs_.size() > 1) {
      outputs_[1]->set(stream_output_, true);
    }
    return base::kStatusCodeOk;
  };

  using dag::Node::serialize;
  virtual base::Status serialize(
      rapidjson::Value& json,
      rapidjson::Document::AllocatorType& allocator) override {
    base::Status status = dag::Node::serialize(json, allocator);
    if (status != base::kStatusCodeOk) {
      return status;
    }
    json.AddMember("enable_stream_", enable_stream_, allocator);
    return status;
  }
  using dag::Node::deserialize;
  virtual base::Status deserialize(rapidjson::Value& json) override {
    base::Status status = dag::Node::deserialize(json);
    if (status != base::kStatusCodeOk) {
      return status;
    }
    if (json.HasMember("enable_stream_") && json["enable_stream_"].IsBool()) {
      enable_stream_ = json["enable_stream_"].GetBool();
    }
    return status;
  }

 private:
  // 流式输出配置
  bool enable_stream_ = true;
  // 是否为第一次输出
  bool is_first_ = true;
  // 结果
  tokenizer::TokenizerText* output_text_ = nullptr;
  std::string* stream_output_ = nullptr;
  //
  std::vector<std::string> stop_texts_ = {
      "<|endoftext|>", "<|im_end|>", "</s>", "<|end|>", "<|eot_id|>", "[DONE]"};
};

}  // namespace llm
}  // namespace nndeploy

#endif
