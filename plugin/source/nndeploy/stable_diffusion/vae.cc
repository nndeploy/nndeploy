#include "nndeploy/stable_diffusion/vae.h"

#include "nndeploy/infer/infer.h"
#include "nndeploy/op/op_muls.h"
#include "nndeploy/stable_diffusion/ddim_scheduler.h"
#include "nndeploy/stable_diffusion/utils.h"

namespace nndeploy {
namespace stable_diffusion {

class NNDEPLOY_CC_API ScaleLatentsParam : public base::Param {
 public:
  float vae_scale_factor_ = 0.18215;
  virtual base::Status serialize(
      rapidjson::Value &json, rapidjson::Document::AllocatorType &allocator) {
    json.AddMember("vae_scale_factor_", vae_scale_factor_, allocator);
    return base::kStatusCodeOk;
  }
  virtual base::Status deserialize(rapidjson::Value &json) {
    if (json.HasMember("vae_scale_factor_") &&
        json["vae_scale_factor_"].IsFloat()) {
      vae_scale_factor_ = json["vae_scale_factor_"].GetFloat();
    }
    return base::kStatusCodeOk;
  }
};

class NNDEPLOY_CC_API ScaleLatents : public dag::Node {
 public:
  ScaleLatents(const std::string &name, std::vector<dag::Edge *> inputs,
               std::vector<dag::Edge *> outputs)
      : dag::Node(name, inputs, outputs) {
    key_ = "nndeploy::stable_diffusion::ScaleLatents";
    desc_ = "stable_diffusion scale latents [device::Tensor->device::Tensor]";
    param_ = std::make_shared<ScaleLatentsParam>();
    this->setInputTypeInfo<device::Tensor>();
    this->setOutputTypeInfo<device::Tensor>();
  }

  virtual ~ScaleLatents() {}

  virtual base::Status init() {
    device::Device *device = device::getDefaultHostDevice();
    device::TensorDesc scalar_desc;
    scalar_desc.data_type_ = base::dataTypeOf<float>();
    scalar_desc.data_format_ = base::kDataFormatNC;
    scalar_desc.shape_ = {1};
    scalar_ = new device::Tensor(device, scalar_desc);
    return base::kStatusCodeOk;
  }

  virtual base::Status deinit() {
    if (scalar_ != nullptr) {
      delete scalar_;
    }
    return base::kStatusCodeOk;
  }

  virtual base::Status run() {
    device::Device *device = device::getDefaultHostDevice();
    ScaleLatentsParam *param = (ScaleLatentsParam *)param_.get();
    vae_scale_factor_ = param->vae_scale_factor_;
    float scale_factor = 1 / vae_scale_factor_;
    scalar_->set(scale_factor);

    device::Tensor *latents = this->getInput(0)->getTensor(this);
    device::Tensor *latents_scale =
        this->getOutput(0)->create(device, latents->getDesc());

    op::muls(scalar_, latents, latents_scale);

    this->getOutput(0)->notifyWritten(latents_scale);

    return base::kStatusCodeOk;
  }

 private:
  float vae_scale_factor_ = 0.18215;
  device::Tensor *scalar_;
};

class NNDEPLOY_CC_API VaeGraph : public dag::Graph {
 public:
  VaeGraph(const std::string &name, std::vector<dag::Edge *> inputs,
           std::vector<dag::Edge *> outputs)
      : dag::Graph(name, inputs, outputs) {
    key_ = "nndeploy::stable_diffusion::VaeGraph";
    desc_ = "vae decoder graph [denoise latent -> device::Tensor]";
    this->setInputTypeInfo<device::Tensor>();
    this->setOutputTypeInfo<device::Tensor>();
    scale_ = dynamic_cast<ScaleLatents *>(
        this->createNode<ScaleLatents>("scale_latents"));
    infer_ = dynamic_cast<infer::Infer *>(
        this->createNode<infer::Infer>("vae_infer"));
  }

  virtual ~VaeGraph() {}

  virtual base::Status setNodeExternalParam() {
    ScaleLatentsParam *scale_param =
        dynamic_cast<ScaleLatentsParam *>(scale_->getParam());
    ScaleLatentsParam *scale_param_ = dynamic_cast<ScaleLatentsParam *>(
        getExternalParam("scale_param").get());
    *scale_param = *scale_param_;

    inference::InferenceParam *infer_param =
        dynamic_cast<inference::InferenceParam *>(infer_->getParam());
    inference::InferenceParam *infer_param_ =
        dynamic_cast<inference::InferenceParam *>(
            getExternalParam("vae_infer_param").get());
    *infer_param = *infer_param_;
    return base::kStatusCodeOk;
  }

  base::Status make(const dag::NodeDesc &scale_desc,
                    const dag::NodeDesc &infer_desc,
                    base::InferenceType inference_type) {
    this->setNodeDesc(scale_, scale_desc);
    this->setNodeDesc(infer_, infer_desc);
    base::Status status = infer_->setInferenceType(inference_type);
    if (status != base::kStatusCodeOk) {
      NNDEPLOY_LOGE("Failed to set inference type");
      return status;
    }
    this->setNodeExternalParam();
    return base::kStatusCodeOk;
  }

 private:
  dag::Node *scale_ = nullptr;
  infer::Infer *infer_ = nullptr;
};

dag::Graph *createVAEGraph(const std::string &name, dag::Edge *latents,
                           dag::Edge *output,
                           base::InferenceType inference_type,
                           std::vector<base::Param *> &param) {
  Text2ImageParam *text2image_param = (Text2ImageParam *)param[0];
  auto infer_param = std::make_shared<inference::InferenceParam>();
  infer_param->device_type_ = text2image_param->device_type_;
  infer_param->model_type_ = text2image_param->model_type_;
  infer_param->is_path_ = text2image_param->is_path_;
  std::vector<std::string> onnx_path = {text2image_param->model_value_[3]};
  infer_param->model_value_ = onnx_path;

  DDIMSchedulerParam *scheduler_param = (DDIMSchedulerParam *)param[1];
  auto scale_param = std::make_shared<ScaleLatentsParam>();
  scale_param->vae_scale_factor_ = scheduler_param->vae_scale_factor_;

  VaeGraph *vae_graph = new VaeGraph(name, {latents}, {output});
  dag::NodeDesc scale_desc("scale_latents", {latents->getName()}, {"vae_in"});
  dag::NodeDesc infer_desc("vae_infer", {"vae_in"}, {output->getName()});
  vae_graph->setExternalParam("scale_param", scale_param);
  vae_graph->setExternalParam("vae_infer_param", infer_param);
  vae_graph->make(scale_desc, infer_desc, inference_type);
  return vae_graph;
}

REGISTER_NODE("nndeploy::stable_diffusion::ScaleLatents", ScaleLatents);
REGISTER_NODE("nndeploy::stable_diffusion::VaeGraph", VaeGraph);

}  // namespace stable_diffusion
}  // namespace nndeploy