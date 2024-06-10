
#include "nndeploy/model/stable_diffusion/clip.h"

#include "nndeploy/model/convert_to.h"
#include "nndeploy/model/infer.h"
#include "nndeploy/model/tokenizer/clip_tokenizer.h"


namespace nndeploy {
namespace model {

/**
 * @brief
 *
 * @param name
 * @param input
 * @param output
 * @param param
 * @return dag::Graph*
 * input(prompt + negative prompt) -> cliptokenizer(yes or no) -> CLIP(batch) ->
 * output(text_embeddings)
 */
dag::Graph *createCLIPGraph(const std::string &name, dag::Edge *input,
                            dag::Edge *output,
                            base::InferenceType inference_type,
                            std::vector<base::Param *> &param) {
  dag::Graph *graph = new dag::Graph(name, input, output);

  /**
   * @brief
   * yes or no,
   * 目前这个阶段暂时没做CLIPTokenizer，所以input就是input_ids，会在内部做一个拷贝操作
   */
  // dag::Node *clip_clip_tokenizer =
  //     graph->createNode<CLIPTokenizer>("clip_clip_tokenizer", input,
  //     "input_ids");
  // tokenizer->setParam(param[0]);
  dag::Node *clip_convert_to_pre =
      graph->createNode<ConvertTo>("clip_convert_to_pre", input, "input_ids");
  clip_convert_to_pre->setParam(param[0]);

  /**
   * @brief
   * batch = 2， 多batch的推理
   */
  dag::Node *clip_clip_infer = graph->createInfer<Infer>(
      "clip_clip_infer", inference_type, "input_ids", "text_embeddings");
  clip_clip_infer->setParam(param[1]);

  dag::Node *clip_convert_to_post = graph->createNode<ConvertTo>(
      "clip_convert_to_post", "text_embeddings", output);
  clip_convert_to_post->setParam(param[2]);

  return graph;
}

}  // namespace model
}  // namespace nndeploy
