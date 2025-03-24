#include "nndeploy/stable_diffusion/denoise.h"

namespace nndeploy {
namespace stable_diffusion {

dag::Graph *createDenoiseGraph(const std::string &name,
                               dag::Edge *text_embeddings, dag::Edge *output,
                               SchedulerType scheduler_type,
                               base::InferenceType inference_type,
                               std::vector<base::Param *> &param) {
  dag::Graph *graph = new dag::Graph(name, {text_embeddings}, {output});
  return graph;
}

}  // namespace stable_diffusion
}  // namespace nndeploy