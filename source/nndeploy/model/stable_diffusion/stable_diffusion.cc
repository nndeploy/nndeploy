
#include "nndeploy/model/stable_diffusion/stable_diffusion.h"

#include "nndeploy/model/stable_diffusion/clip.h"
#include "nndeploy/model/stable_diffusion/scheduler.h"
#include "nndeploy/model/infer.h"

// TODO: Edge要支持能够支持同名的情况
// TODO: Node的名字能够支持同名的情况

namespace nndeploy {
namespace model {

/**
 * @brief Create a Stable Diffusion Text2Image Graph object
 *
 * @param name
 * @param input
 * @param output
 * @param param
 * @return dag::Graph*
 * input(prompt + negative prompt) -> clip -> scheduler_unet -> vae ->
 * output(image)
 */
dag::Graph *createStableDiffusionText2ImageGraph(
    const std::string &name, dag::Edge *input, dag::Edge *output,
    base::InferenceType clip_inference_type, SchedulerType scheduler_type,
    base::InferenceType unet_inference_type,
    base::InferenceType vae_inference_type, std::vector<base::Param *> &param) {
  // graph
  dag::Graph *graph = new dag::Graph(name, input, output);

  // clip
  dag::Edge *encoder_hidden_states = graph->createEdge("encoder_hidden_states");
  dag::Graph *sd_clip = createCLIPGraph("sd_clip", input, encoder_hidden_states,
                                        clip_inference_type, param);
  graph->addNode(sd_clip, false);

  // scheduler_unet
  dag::Edge *latent = graph->createEdge("latent");
  std::vector<base::Param *>::iterator it = param.begin() + 3;
  std::vector<base::Param *> schedule_unet_param(it, it+ 2);
  dag::Graph *sd_scheduler_unet = createSchedulerUNetGraph(
      "sd_scheduler_unet", encoder_hidden_states, latent, scheduler_type,
      unet_inference_type, schedule_unet_param);
  graph->addNode(sd_scheduler_unet, false);

  // vae
  dag::Node *sd_vae =
      graph->createInfer<Infer>("sd_vae", vae_inference_type, latent, output);
  sd_vae->setParam(param[5]);

  return graph;
}

}  // namespace model
}  // namespace nndeploy