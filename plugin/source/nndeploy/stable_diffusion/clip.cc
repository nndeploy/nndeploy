#include "nndeploy/stable_diffusion/clip.h"

namespace nndeploy {
namespace stable_diffusion {

dag::Graph *createCLIPGraph(const std::string &name, dag::Edge *prompt,
                            dag::Edge *negative_prompt, dag::Edge *output,
                            base::InferenceType inference_type,
                            std::vector<base::Param *> &param) {
  dag::Graph *graph = new dag::Graph(name, {prompt, negative_prompt}, {output});
  return graph;
}

}  // namespace stable_diffusion
}  // namespace nndeploy