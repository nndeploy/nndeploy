#include "nndeploy/stable_diffusion/vae.h"

#include "nndeploy/infer/infer.h"
#include "nndeploy/op/op_muls.h"

namespace nndeploy {
namespace stable_diffusion {

class NNDEPLOY_CC_API ScaleLatentsNode : public dag::Node {
 public:
  ScaleLatentsNode(const std::string &name, std::vector<dag::Edge *> inputs,
                   std::vector<dag::Edge *> outputs)
      : dag::Node(name, inputs, outputs) {}
  virtual ~ScaleLatentsNode() {}
  virtual base::Status run() {
    float scale_factor = 1 / vae_scale_factor;
    device::Device *device = device::getDefaultHostDevice();
    device::TensorDesc scalar_desc;
    scalar_desc.data_type_ = base::dataTypeOf<float>();
    scalar_desc.data_format_ = base::kDataFormatNC;
    scalar_desc.shape_ = {1};
    device::Tensor *scalar = new device::Tensor(device, scalar_desc);
    scalar->set(scale_factor);

    device::Tensor *latents = this->getInput(0)->getTensor(this);
    device::Tensor *latents_scale =
        new device::Tensor(device, latents->getDesc());

    op::muls(scalar, latents, latents_scale);

    this->getOutput(0)->set(latents_scale, 0);
    return base::kStatusCodeOk;
  }

 private:
  float vae_scale_factor = 0.18215;
};

dag::Graph *createVAEGraph(const std::string &name, dag::Edge *latents,
                           dag::Edge *output,
                           base::InferenceType inference_type,
                           std::vector<base::Param *> &param) {
  dag::Graph *graph = new dag::Graph(name, {latents}, {output});
  dag::Edge *model_input = graph->createEdge("vae_input");
  dag::Node *scale_node = graph->createNode<ScaleLatentsNode>(
      "scale_latents", {latents}, {model_input});
  dag::Node *vae_node = graph->createInfer<infer::Infer>(
      "vae_infer", inference_type, {model_input}, {output});
  inference::InferenceParam *infer_param = new inference::InferenceParam();
  infer_param->device_type_ = base::kDeviceTypeCodeCpu;
  infer_param->model_type_ = base::kModelTypeOnnx;
  infer_param->is_path_ = true;
  std::vector<std::string> onnx_path = {
      "/home/lds/stable-diffusion.onnx/models/vae_decoder/model.onnx"};
  infer_param->model_value_ = onnx_path;
  vae_node->setParam(infer_param);

  return graph;
}

}  // namespace stable_diffusion
}  // namespace nndeploy