#include "nndeploy/stable_diffusion/vae.h"

namespace nndeploy {
namespace stable_diffusion {

dag::Graph *createVAEGraph(const std::string &name, dag::Edge *latents,
                           dag::Edge *output,
                           base::InferenceType inference_type,
                           std::vector<base::Param *> &param) {
  dag::Graph *graph = new dag::Graph(name, {latents}, {output});
  return graph;
}

}  // namespace stable_diffusion
}  // namespace nndeploy