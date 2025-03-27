#include "nndeploy/stable_diffusion/vae.h"

#include "nndeploy/infer/infer.h"

namespace nndeploy {
namespace stable_diffusion {

dag::Graph *createVAEGraph(const std::string &name, dag::Edge *latents,
                           dag::Edge *output,
                           base::InferenceType inference_type,
                           std::vector<base::Param *> &param) {
  dag::Graph *graph = new dag::Graph(name, {latents}, {output});
  graph->createInfer<infer::Infer>("vae_infer", inference_type, {latents},
                                   {output});
  inference::InferenceParam *infer_param = new inference::InferenceParam();
  infer_param->device_type_ = base::kDeviceTypeCodeCpu;
  infer_param->model_type_ = base::kModelTypeOnnx;
  infer_param->is_path_ = true;
  std::vector<std::string> onnx_path = {
      "/home/lds/stable-diffusion.onnx/models/vae_decoder"};
  infer_param->model_value_ = onnx_path;
  graph->setParam(infer_param);

  return graph;
}

}  // namespace stable_diffusion
}  // namespace nndeploy