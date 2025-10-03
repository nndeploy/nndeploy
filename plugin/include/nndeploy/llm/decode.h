
// 将 tokenizer + embedding + infer + sample 封装成一个循环图

#ifndef _NNDEPLOY_LLM_DECODE_H_
#define _NNDEPLOY_LLM_DECODE_H_

#include "nndeploy/dag/graph.h"
#include "nndeploy/dag/loop.h"
#include "nndeploy/llm/embedding.h"
#include "nndeploy/llm/llm_infer.h"
#include "nndeploy/llm/prompt.h"
#include "nndeploy/llm/sample.h"
#include "nndeploy/llm/stream_out.h"
#include "nndeploy/tokenizer/tokenizer.h"

namespace nndeploy {
namespace llm {

class Decode : public dag::Loop {
 public:
  Decode(const std::string& name, std::vector<dag::Edge*> inputs,
         std::vector<dag::Edge*> outputs);
  virtual ~Decode();

  virtual base::Status make(const dag::NodeDesc& infer,
                            const dag::NodeDesc& sample,
                            const dag::NodeDesc& tokenizer,
                            const dag::NodeDesc& stream_out);

  virtual base::Status initEnd() override;
  virtual base::Status iterAfter() override;

  virtual std::vector<dag::Edge*> forward(dag::Edge* input) override;

  void getStopTokens(std::string& token_file);
  virtual int loops() override;
  virtual bool isStop();
  virtual bool isStopTokens();
  virtual bool isStopTexts();

  virtual base::Status serialize(
      rapidjson::Value& json,
      rapidjson::Document::AllocatorType& allocator) override;
  virtual base::Status deserialize(rapidjson::Value& json) override;

 private:
  LlmInfer* decode_infer_node_;
  dag::Node* decode_sampler_node_;
  dag::Node* decode_token_node_;
  StreamOut* stream_out_node_;

  bool is_first_ = true;
  int max_seq_len_ = std::numeric_limits<int>::max();

  std::string tokenizer_txt_ = "";
  std::vector<int> stop_tokens_;
  std::vector<int> special_tokens_;

  std::vector<std::string> stop_texts_ = {
      "<|endoftext|>", "<|im_end|>", "</s>", "<|end|>", "<|eot_id|>", "[DONE]"};
  std::vector<std::string> special_texts_;
};

}  // namespace llm
}  // namespace nndeploy

#endif
