#include "nndeploy/llm/decode.h"

#include "nndeploy/tokenizer/tokenizer_cpp/tokenizer_cpp.h"

namespace nndeploy {
namespace llm {

Decode::Decode(const std::string& name, std::vector<dag::Edge*> inputs,
               std::vector<dag::Edge*> outputs)
    : dag::Loop(name, inputs, outputs) {
  key_ = "nndeploy::llm::Decode";
  desc_ = "Decode: Decode pipeline";

  this->setInputTypeInfo<tokenizer::TokenizerIds>("input_tokens");
  this->setOutputTypeInfo<tokenizer::TokenizerText>("output_tokens");

  decode_infer_node_ = this->createNode<llm::LlmInfer>("decode_infer");
  decode_sampler_node_ = this->createNode<Sampler>("decode_sampler_node");
  decode_token_node_ =
      this->createNode<tokenizer::TokenizerDecodeCpp>("token_node");
}

Decode::~Decode() {}

base::Status Decode::make(const dag::NodeDesc& infer,
                          const dag::NodeDesc& sample,
                          const dag::NodeDesc& tokenizer) {
  base::Status status = base::kStatusCodeOk;
  status = this->setNodeDesc(decode_infer_node_, infer);
  NNDEPLOY_RETURN_ON_NEQ(status, base::kStatusCodeOk,
                         "decode_infer_node_ set node desc failed!");
  status = this->setNodeDesc(decode_sampler_node_, sample);
  NNDEPLOY_RETURN_ON_NEQ(status, base::kStatusCodeOk,
                         "decode_sampler_node_ set node desc failed!");
  status = this->setNodeDesc(decode_token_node_, tokenizer);
  NNDEPLOY_RETURN_ON_NEQ(status, base::kStatusCodeOk,
                         "decode_token_node_ set node desc failed!");
  return base::kStatusCodeOk;
}

int Decode::loops() { return 1; }

std::vector<dag::Edge*> Decode::forward(dag::Edge* input) {
  std::vector<dag::Edge*> output;
  for (int i = 0; i < this->loops(); i++) {
    output = (*decode_infer_node_)(input);   
    output = (*decode_sampler_node_)(output);
    output = (*decode_token_node_)(output);
  }
  return output;
}

REGISTER_NODE("nndeploy::llm::Decode", Decode);

}  // namespace llm
}  // namespace nndeploy
