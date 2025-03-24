#include "nndeploy/stable_diffusion/pipeline.h"

#include "nndeploy/stable_diffusion/clip.h"
#include "nndeploy/stable_diffusion/denoise.h"
#include "nndeploy/stable_diffusion/vae.h"

namespace nndeploy {
namespace stable_diffusion {

dag::Graph *createStableDiffusionText2ImageGraph(
    const std::string name, dag::Edge *prompt, dag::Edge *negative_prompt,
    dag::Edge *output, base::InferenceType clip_inference_type,
    base::InferenceType unet_inference_type,
    base::InferenceType vae_inference_type, SchedulerType scheduler_type,
    std::vector<base::Param *> &param) {
  dag::Graph *graph = new dag::Graph(name, {prompt, negative_prompt}, {output});

  dag::Edge *text_embeddings = graph->createEdge("text_embeddings");
  dag::Graph *clip_graph =
      createCLIPGraph("clip", prompt, negative_prompt, text_embeddings,
                      clip_inference_type, param);
  graph->addNode(clip_graph, false);

  dag::Edge *latents = graph->createEdge("latents");
  dag::Graph *denoise_graph =
      createDenoiseGraph("denoise_ddim", text_embeddings, latents,
                         scheduler_type, unet_inference_type, param);
  graph->addNode(denoise_graph, false);

  dag::Node *vae_graph =
      createVAEGraph("vae", latents, output, vae_inference_type, param);
  graph->addNode(vae_graph, false);

  return graph;
}

}  // namespace stable_diffusion
}  // namespace nndeploy