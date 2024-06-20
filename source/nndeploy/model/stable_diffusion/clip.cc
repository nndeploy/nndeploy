
#include "nndeploy/model/stable_diffusion/clip.h"

#include "nndeploy/model/convert_to.h"
#include "nndeploy/model/infer.h"
#include "nndeploy/model/tokenizer/clip_tokenizer.h"

namespace nndeploy {
namespace model {

dag::Graph *createCLIPGraph(const std::string &name, dag::Edge *prompt,
                            dag::Edge *negative_prompt, dag::Edge *output,
                            base::InferenceType inference_type,
                            std::vector<base::Param *> &param) {
  dag::Graph *graph = new dag::Graph(name, {prompt, negative_prompt}, {output});

  //   /**
  //    * @brief tokenizer
  //    * prompt or negative_prompt
  //    */
  //   dag::Node *tokenizer_prompt = graph->createNode<CLIPTokenizer>(
  //       "tokenizer_prompt", prompt, "prompt_ids");
  //   tokenizer_prompt->setParam(param[0]);
  //   dag::Node *tokenizer_negative_prompt = graph->createNode<CLIPTokenizer>(
  //       "tokenizer_negative_prompt", negative_prompt, "negative_prompt_ids");
  //   tokenizer_prompt->setParam(param[1]);

  //   /**
  //    * @brief TensorConcat
  //    */
  //   dag::Node *tensor_concat = graph->createNode<TensorConcat>(
  //       "tensor_concat", inference_type, {"prompt_ids",
  //       "negative_prompt_ids"},
  //       {"input_ids"});
  //   tensor_concat->setParam(param[2]);

  /**
   * @brief createInfer
   * batch = 2， 多batch的推理
   */
  dag::Node *infer =
      graph->createInfer<Infer>("infer", inference_type, "input_ids", output);
  infer->setParam(param[3]);

  return graph;
}

}  // namespace model
}  // namespace nndeploy
