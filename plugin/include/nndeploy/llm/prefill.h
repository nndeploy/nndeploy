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
  virtual ~Prefill();

  virtual base::Status defaultParam();

  virtual base::Status make(const dag::NodeDesc& tokenizer,
                            const dag::NodeDesc& embedding,
                            const dag::NodeDesc& infer,
                            const dag::NodeDesc& sample);

  virtual base::Status serialize(rapidjson::Value& json,
                                 rapidjson::Document::AllocatorType& allocator);
  virtual base::Status deserialize(rapidjson::Value& json);

  virtual std::vector<dag::Edge*> forward(dag::Edge* input) override;

 private:
  dag::Node* prefill_token_node_;
  dag::Node* prefill_embedding_node_;
  dag::Node* prefill_infer_node_;
  dag::Node* prefill_sample_node_;
};

}  // namespace llm
}  // namespace nndeploy

#endif