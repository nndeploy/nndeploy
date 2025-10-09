// 将tokenizer + embedding + infer + sample 封装成一个图

#ifndef _NNDEPLOY_LLM_PREFILL_H_
#define _NNDEPLOY_LLM_PREFILL_H_

#include "nndeploy/dag/graph.h"
#include "nndeploy/llm/embedding.h"
#include "nndeploy/llm/llm_infer.h"
#include "nndeploy/llm/prompt.h"
#include "nndeploy/llm/sample.h"
#include "nndeploy/tokenizer/tokenizer.h"

namespace nndeploy {
namespace llm {

class Prefill : public dag::Graph {
 public:
  Prefill(const std::string& name, std::vector<dag::Edge*> inputs,
          std::vector<dag::Edge*> outputs);
  Prefill(const std::string& name);
  virtual ~Prefill();

  virtual base::Status make(const dag::NodeDesc& tokenizer,
                            const dag::NodeDesc& infer,
                            const dag::NodeDesc& sample);

  virtual std::vector<dag::Edge*> forward(dag::Edge* input) override;

 private:
  dag::Node* prefill_token_node_ = nullptr;
  llm::LlmInfer* prefill_infer_node_ = nullptr;
  dag::Node* prefill_sampler_node_ = nullptr;
};

}  // namespace llm
}  // namespace nndeploy

#endif