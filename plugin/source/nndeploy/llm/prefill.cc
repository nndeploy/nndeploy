#include "nndeploy/llm/prefill.h"

#include "nndeploy/tokenizer/tokenizer_cpp/tokenizer_cpp.h"

namespace nndeploy {
namespace llm {

Prefill::Prefill(const std::string& name, std::vector<dag::Edge*> inputs,
                 std::vector<dag::Edge*> outputs)
    : dag::Graph(name, inputs, outputs) {
  key_ = "nndeploy::llm::Prefill";
  desc_ = "Prefill: Prefill pipeline";

  this->setInputTypeInfo<tokenizer::TokenizerText>("input_text");
  this->setOutputTypeInfo<tokenizer::TokenizerIds>("output_tokens");

  prefill_token_node_ =
      this->createNode<tokenizer::TokenizerEncodeCpp>("token_node");
  prefill_infer_node_ = dynamic_cast<llm::LlmInfer*>(this->createNode<llm::LlmInfer>("prefill_infer"));
  prefill_infer_node_->setPrefill(true);
  prefill_sampler_node_ =
      this->createNode<Sampler>("prefill_sampler_node");
}
Prefill::Prefill(const std::string& name) : dag::Graph(name) {
  key_ = "nndeploy::llm::Prefill";
  desc_ = "Prefill: Prefill pipeline";

  this->setInputTypeInfo<tokenizer::TokenizerText>("input_text");
  this->setOutputTypeInfo<tokenizer::TokenizerIds>("output_tokens");

  prefill_token_node_ =
      this->createNode<tokenizer::TokenizerEncodeCpp>("token_node");
  prefill_infer_node_ = dynamic_cast<llm::LlmInfer*>(this->createNode<llm::LlmInfer>("prefill_infer"));
  prefill_infer_node_->setPrefill(true);
  prefill_sampler_node_ =
      this->createNode<Sampler>("prefill_sampler_node");
}
Prefill::~Prefill() {}

base::Status Prefill::make(const dag::NodeDesc& tokenizer,
                           const dag::NodeDesc& infer,
                           const dag::NodeDesc& sampler) {
  base::Status status = base::kStatusCodeOk;
  status = this->setNodeDesc(prefill_token_node_, tokenizer);
  NNDEPLOY_RETURN_ON_NEQ(status, base::kStatusCodeOk,
                         "prefill_token_node_ set node desc failed!");
  status = this->setNodeDesc(prefill_infer_node_, infer);
  NNDEPLOY_RETURN_ON_NEQ(status, base::kStatusCodeOk,
                         "prefill_infer_node_ set node desc failed!");
  status = this->setNodeDesc(prefill_sampler_node_, sampler);
  NNDEPLOY_RETURN_ON_NEQ(status, base::kStatusCodeOk,
                         "prefill_sampler_node_ set node desc failed!");
  return base::kStatusCodeOk;
}

std::vector<dag::Edge*> Prefill::forward(dag::Edge* input) {
  std::vector<dag::Edge*> output = (*prefill_token_node_)(input);
  output = (*prefill_infer_node_)(output);
  output = (*prefill_sampler_node_)(output);
  return output;
}

REGISTER_NODE("nndeploy::llm::Prefill", Prefill);

}  // namespace llm
}  // namespace nndeploy