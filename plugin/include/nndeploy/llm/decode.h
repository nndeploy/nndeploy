
// 将 tokenizer + embedding + infer + sample 封装成一个循环图

#ifndef _NNDEPLOY_LLM_DECODE_H_
#define _NNDEPLOY_LLM_DECODE_H_

#include "nndeploy/dag/graph.h"
#include "nndeploy/dag/loop.h"
#include "nndeploy/llm/embedding.h"
#include "nndeploy/llm/llm_infer.h"
#include "nndeploy/llm/prompt.h"
#include "nndeploy/llm/sample.h"
#include "nndeploy/tokenizer/tokenizer.h"

namespace nndeploy {
namespace llm {

class Decode : public dag::Loop {
 public:
  Decode(const std::string& name, std::vector<dag::Edge*> inputs,
         std::vector<dag::Edge*> outputs);
  virtual ~Decode();

  virtual base::Status make(const dag::NodeDesc& tokenizer,
                            const dag::NodeDesc& infer,
                            const dag::NodeDesc& sample);

  virtual int loops() override;

  //   virtual base::Status run() override;

  virtual std::vector<dag::Edge*> forward(dag::Edge* input) override;

 private:
  dag::Node* decode_token_node_;
  dag::Node* decode_infer_node_;
  dag::Node* decode_sampler_node_;
};

}  // namespace llm
}  // namespace nndeploy

#endif
