#include "nndeploy/stable_diffusion/denoise.h"

#include "nndeploy/infer/infer.h"
#include "nndeploy/op/op_add.h"
#include "nndeploy/op/op_concat.h"
#include "nndeploy/op/op_muls.h"
#include "nndeploy/op/op_split.h"
#include "nndeploy/op/op_sub.h"
#include "nndeploy/stable_diffusion/ddim_scheduler.h"
#include "nndeploy/stable_diffusion/scheduler.h"

namespace nndeploy {
namespace stable_diffusion {

class DenoiseGraph : public dag::Loop {
 public:
  DenoiseGraph(const std::string name, SchedulerType scheduler_type,
               std::vector<dag::Edge *> inputs,
               std::vector<dag::Edge *> outputs)
      : Loop(name, inputs, outputs), scheduler_type_(scheduler_type) {
    param_ = std::make_shared<DDIMSchedulerParam>();
    scheduler_ = createScheduler(scheduler_type_);
  }

  virtual ~DenoiseGraph() {}

  virtual base::Status init() {
    base::Status status = base::kStatusCodeOk;
    scheduler_param_ = (SchedulerParam *)(param_.get());
    status = scheduler_->init(scheduler_param_);
    NNDEPLOY_RETURN_ON_NEQ(status, base::kStatusCodeOk, "init failed!");
    return status;
  }

  virtual base::Status deinit() {
    base::Status status = base::kStatusCodeOk;
    status = scheduler_->deinit();
    NNDEPLOY_RETURN_ON_NEQ(status, base::kStatusCodeOk, "deinit failed!");
    return status;
  }

  virtual int loops() { return scheduler_param_->num_inference_steps_; }

  virtual base::Status run() {
    base::Status status = base::kStatusCodeOk;
    setRunningFlag(true);

    bool do_classifier_free_guidance = (guidance_ > 1.0) ? true : false;

    int index = this->getInput(0)->getIndex(this);

    SchedulerParam *scheduler_param = (SchedulerParam *)(param_.get());

    // initialize latents
    device::Device *device = device::getDefaultHostDevice();
    device::TensorDesc latent_desc;
    latent_desc.data_type_ = base::dataTypeOf<float>();
    latent_desc.data_format_ = base::kDataFormatNCHW;
    latent_desc.shape_ = {1, 4, 64, 64};
    device::Tensor *latent = new device::Tensor(device, latent_desc, "latent");
    std::mt19937 generator;
    initializeLatents(generator, init_noise_sigma_, latent);
    device::Tensor *latent_copy = latent->clone();

    // classifier free guidance
    dag ::Edge *sample = this->getEdge("sample");
    device::Tensor *latents = nullptr;
    device::TensorDesc latent_model_input_desc;
    if (do_classifier_free_guidance) {
      latent_model_input_desc.data_type_ = base::dataTypeOf<float>();
      latent_model_input_desc.data_format_ = base::kDataFormatNCHW;
      latent_model_input_desc.shape_ = {2, 4, 64, 64};
      latents = sample->create(device, latent_model_input_desc, index);
    } else {
      latent_model_input_desc.data_type_ = base::dataTypeOf<float>();
      latent_model_input_desc.data_format_ = base::kDataFormatNCHW;
      latent_model_input_desc.shape_ = {1, 4, 64, 64};
      latents = sample->create(device, latent_model_input_desc, index);
    }

    // create timestep infer tensor
    dag::Edge *timestep = this->getEdge("timestep");
    device::TensorDesc timestep_desc;
    timestep_desc.data_type_ = base::dataTypeOf<float>();
    timestep_desc.data_format_ = base::kDataFormatNC;
    timestep_desc.shape_.emplace_back(1);
    device::Tensor *timestep_tensor =
        timestep->create(device, timestep_desc, index);

    // create unet output tensor
    device::TensorDesc noise_desc;
    noise_desc.data_type_ = base::dataTypeOf<float>();
    noise_desc.data_format_ = base::kDataFormatNCHW;
    noise_desc.shape_ = {1, 4, 64, 64};
    device::Tensor *noise_pred = new device::Tensor(device, noise_desc);
    device::Tensor *noise_pred_uncond = new device::Tensor(device, noise_desc);
    device::Tensor *noise_sub = new device::Tensor(device, noise_desc);
    device::Tensor *noise_muls = new device::Tensor(device, noise_desc);

    device::TensorDesc scalar_desc;
    scalar_desc.data_type_ = base::dataTypeOf<float>();
    scalar_desc.data_format_ = base::kDataFormatNC;
    scalar_desc.shape_.emplace_back(1);
    device::Tensor *scalar = new device::Tensor(device, scalar_desc);
    scalar->set(guidance_);

    scheduler_->setTimesteps();
    std::vector<int> timesteps = scheduler_->getTimestep();
    for (const auto &val : timesteps) {
      if (do_classifier_free_guidance) {
        op::concat({latent_copy, latent_copy}, 0, latents);
      } else {
        latents = latent_copy->clone();
      }

      timestep_tensor->set((float)val);

      status = executor_->run();
      NNDEPLOY_RETURN_ON_NEQ(status, base::kStatusCodeOk, "unet run failed.\n");

      dag::Edge *unet_outputs = this->getEdge("unet_outputs");
      device::Tensor *unet_output_tensor = unet_outputs->getTensor(this);
      if (do_classifier_free_guidance) {
        std::vector<device::Tensor *> outputs = {noise_pred_uncond, noise_pred};
        std::shared_ptr<ir::SplitParam> split_param =
            std::make_shared<ir::SplitParam>();
        split_param->axis_ = 0;
        split_param->num_outputs_ = 2;
        op::split(unet_output_tensor, split_param, outputs);
        op::sub(noise_sub, noise_pred, noise_pred_uncond);
        op::muls(scalar, noise_muls, noise_sub);
        op::add(noise_pred, noise_muls, noise_pred_uncond);
      } else {
        noise_pred = unet_output_tensor;
      }
      scheduler_->step(noise_pred, timestep_tensor, latent, latent_copy);
    }
    return status;
  }

 private:
  SchedulerType scheduler_type_ = kSchedulerTypeNotSupport;
  Scheduler *scheduler_ = nullptr;
  SchedulerParam *scheduler_param_ = nullptr;

  float init_noise_sigma_ = 1.0f;
  float guidance_ = 1.0;
};

dag::Graph *createDenoiseGraph(const std::string &name,
                               dag::Edge *text_embeddings, dag::Edge *latents,
                               SchedulerType scheduler_type,
                               base::InferenceType inference_type,
                               std::vector<base::Param *> &param) {
  DenoiseGraph *denoise_graph =
      new DenoiseGraph(name, scheduler_type, {text_embeddings}, {latents});

  dag::Edge *sample = denoise_graph->createEdge("sample");
  dag::Edge *timestep = denoise_graph->createEdge("timestep");
  dag::Edge *unet_output = denoise_graph->createEdge("unet_outputs");
  dag::Node *unet = denoise_graph->createInfer<infer::Infer>(
      "unet", inference_type, {sample, timestep, text_embeddings},
      {unet_output});
  inference::InferenceParam *infer_param = new inference::InferenceParam();
  infer_param->device_type_ = base::kDeviceTypeCodeCpu;
  infer_param->model_type_ = base::kModelTypeOnnx;
  infer_param->is_path_ = true;
  std::vector<std::string> onnx_path = {
      "/home/lds/stable-diffusion.onnx/models/unet/"};
  infer_param->model_value_ = onnx_path;
  unet->setParam(infer_param);

  return denoise_graph;
}

}  // namespace stable_diffusion
}  // namespace nndeploy