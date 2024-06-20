
#include "nndeploy/model/infer.h"
#include "nndeploy/model/stable_diffusion/clip.h"
#include "nndeploy/model/stable_diffusion/scheduler.h"
#include "nndeploy/model/stable_diffusion/stable_diffusion.h"

// TODO: Edge要支持能够支持同名的情况
// TODO: Node的名字能够支持同名的情况

namespace nndeploy {
namespace model {

class Text2ImagesSchedulerUNet : public dag::Loop {
 public:
  Text2ImagesSchedulerUNet(const std::string &name,
                           SchedulerType scheduler_type,
                           std::vector<dag::Edge *> &inputs,
                           std::vector<dag::Edge *> &outputs)
      : Loop(name, inputs, outputs), scheduler_type_(scheduler_type) {
    param_ = std::make_shared<SchedulerParam>();
    scheduler_ = createScheduler(scheduler_type_);
    scheduler_->setParam((SchedulerParam *)param_.get());
  }
  virtual ~Text2ImagesSchedulerUNet() { delete scheduler_; }

  virtual base::Status init() {
    base::Status status = base::kStatusCodeOk;

    // # this schedule is very specific to the latent diffusion model.
    // ## 计算betas，它们是方差的平方根，从beta_start的平方根到beta_end的平方根
    SchedulerParam *scheduler_param = (SchedulerParam *)(param_.get());
    std::vector<float> betas;
    betas.resize(scheduler_param->num_train_timesteps_);
    scheduler_->customLinspace(std::sqrtf(scheduler_param->beta_start_),
                               std::sqrtf(scheduler_param->beta_end_),
                               scheduler_param->num_train_timesteps_, betas);
    // ## 计算alphas，它们是1减去beta的平方
    std::vector<float> alphas(scheduler_param->num_train_timesteps_, 0.0f);
    for (int i = 0; i < scheduler_param->num_train_timesteps_; i++) {
      alphas[i] = 1 - betas[i] * betas[i];
    }
    // ## alphas_cumprod_
    alphas_cumprod_.resize(scheduler_param->num_train_timesteps_, 0.0f);
    alphas_cumprod_[0] = alphas[0];
    for (int i = 1; i < scheduler_param->num_train_timesteps_; i++) {
      alphas_cumprod_[i] = alphas_cumprod_[i - 1] * alphas[i];
    }
    // # standard deviation of the initial noise distribution
    init_noise_sigma_ = 1.0f;

    // # At every step in ddim, we are looking into the previous alphas_cumprod
    // For the final step, there is no previous alphas_cumprod because we are
    // already at 0 `set_alpha_to_one` decides whether we set this parameter
    // simply to one or whether we use the final alpha of the "non-previous"
    // one.
    if (scheduler_param->set_alpha_to_one_) {
      final_alpha_cumprod_ = 1.0;
    } else {
      final_alpha_cumprod_ = alphas_cumprod_.front();
    }

    // timesteps_
    timesteps_.resize(scheduler_param->num_train_timesteps_, 0);
    for (int i = 0; i < scheduler_param->num_train_timesteps_; i++) {
      timesteps_[i] = scheduler_param->num_train_timesteps_ - 1 - i;
    }

    status = scheduler_->setTimesteps();
    NNDEPLOY_RETURN_ON_NEQ(status, base::kStatusCodeOk, "setTimesteps failed!");

    status = scheduler_->configure();
    NNDEPLOY_RETURN_ON_NEQ(status, base::kStatusCodeOk, "configure failed!");

    status = Loop::init();
    NNDEPLOY_RETURN_ON_NEQ(status, base::kStatusCodeOk, "Loop::init failed!");

    return status;
  }
  virtual base::Status deinit() {
    base::Status status = base::kStatusCodeOk;
    return status;
  }

  virtual int loops() {
    SchedulerParam *scheduler_param = (SchedulerParam *)(param_.get());
    return scheduler_param->num_inference_steps_;
  }
  virtual base::Status run() {
    base::Status status = base::kStatusCodeOk;

    setRunningFlag(true);

    SchedulerParam *scheduler_param = (SchedulerParam *)(param_.get());

    int size = loops();
    if (size < 1) {
      NNDEPLOY_LOGE("loops size is invalid!\n");
      return base::kStatusCodeErrorInvalidValue;
    }
    device::Device *host_device = device::getDefaultHostDevice();

    // encoder_hidden_states
    dag::Edge *encoder_hidden_states = this->getEdge("encoder_hidden_states");
    device::Tensor *encoder_hidden_states_tensor =
        encoder_hidden_states->getTensor(this);
    int index = encoder_hidden_states->getIndex(this);
    int batch_size = encoder_hidden_states_tensor->getBatch();

    // build sample
    dag::Edge *sample = this->getEdge("sample");
    device::TensorDesc sample_desc;
    sample_desc.data_type_ = base::dataTypeOf<float>();
    sample_desc.data_format_ = base::kDataFormatNCHW;
    sample_desc.shape_.emplace_back(batch_size * 2);
    sample_desc.shape_.emplace_back(scheduler_param->unet_channels_);
    int latent_height = scheduler_param->image_height_ / 8;
    sample_desc.shape_.emplace_back(latent_height);
    int latent_width = scheduler_param->image_width_ / 8;
    sample_desc.shape_.emplace_back(latent_width);
    device::Tensor *sample_tensor =
        sample->create(host_device, sample_desc, index);

    // build timestep
    dag::Edge *timestep = this->getEdge("timestep");
    device::TensorDesc timestep_desc;
    timestep_desc.data_type_ = base::dataTypeOf<float>();
    timestep_desc.data_format_ = base::kDataFormatNC;
    timestep_desc.shape_.emplace_back(batch_size);
    timestep_desc.shape_.emplace_back(1);
    device::Tensor *timestep_tensor =
        timestep->create(host_device, timestep_desc, index);

    // build latent
    device::TensorDesc latent_desc;
    latent_desc.data_type_ = base::dataTypeOf<float>();
    latent_desc.data_format_ = base::kDataFormatNCHW;
    latent_desc.shape_.emplace_back(batch_size);
    latent_desc.shape_.emplace_back(scheduler_param->unet_channels_);
    int latent_height = scheduler_param->image_height_ / 8;
    latent_desc.shape_.emplace_back(latent_height);
    int latent_width = scheduler_param->image_width_ / 8;
    latent_desc.shape_.emplace_back(latent_width);
    device::Tensor *latent = outputs_[0]->create(host_device, latent_desc,
                                                 inputs_[0]->getIndex(this));
    std::mt19937 generator;
    scheduler_->initializeLatents(generator, init_noise_sigma_, latent);

    for (int i = 0; i < size; i++) {
      // op::concat({latent, latent}, 0, sample);
      // status = executor_->run();
      // NNDEPLOY_RETURN_ON_NEQ(status, base::kStatusCodeOk, "executor run
      // failed!");
      // op::split(sample, 0, noise_pred_uncond_, noise_pred_text_)
      // step(noise_pred_uncond_, sample, i, timestep, eta, false, generator,
      //      nullptr);
    }

    // float scale = 1.0f / 0.18215f;
    // op::mul(latents, scale, latents);

    setRunningFlag(false);
    return status;
  }

 private:
  SchedulerType scheduler_type_ = kSchedulerTypeNotSupport;
  Scheduler *scheduler_ = nullptr;
  std::vector<float> alphas_cumprod_;  // alpha的累积乘积
  float final_alpha_cumprod_ = 1.0;
  // standard deviation of the initial noise distribution
  float init_noise_sigma_ = 1.0f;  // 初始噪声的标准差

  std::vector<int64_t> timesteps_;  // 时间步序列
  std::vector<float> variance_;     // 方差
  device::Tensor *noise_pred_uncond_ = nullptr;
  device::Tensor *noise_pred_text_ = nullptr;
};

dag::Graph *createText2ImagesSchedulerUNetGraph(
    const std::string &name, dag::Edge *text_embeddings, dag::Edge *latent,
    SchedulerType scheduler_type, base::InferenceType inference_type,
    std::vector<base::Param *> &param) {
  // inputs
  std::vector<dag::Edge *> inputs;
  inputs.emplace_back(text_embeddings);
  // outputs
  std::vector<dag::Edge *> outputs;
  outputs.emplace_back(latent);
  // scheduler
  Text2ImagesSchedulerUNet *scheduler =
      new Text2ImagesSchedulerUNet(name, scheduler_type, inputs, outputs);
  scheduler->setParam(param[0]);
  // convert_to
  dag::Edge *encoder_hidden_states =
      scheduler->createEdge("encoder_hidden_states");
  // dag::Node *tensor_convert_to = scheduler->createInfer<TensorConvertTo>(
  //     "tensor_convert_to", inference_type, text_embeddings,
  //     encoder_hidden_states);
  // tensor_convert_to->setParam(param[1]);
  // scheduler->addNode(tensor_convert_to, false);

  dag::Edge *sample = scheduler->createEdge("sample");
  dag::Edge *timestep = scheduler->createEdge("timestep");
  dag::Node *unet = scheduler->createInfer<Infer>(
      "unet", inference_type, {sample, timestep, encoder_hidden_states},
      outputs);
  unet->setParam(param[2]);
  scheduler->addNode(unet, false);

  return scheduler;
}

/**
 * @brief Create a Stable Diffusion Text 2 Image Graph object
 *
 * @param name
 * @param prompt
 * @param negative_prompt
 * @param output
 * @param clip_inference_type
 * @param scheduler_type
 * @param unet_inference_type
 * @param vae_inference_type
 * @param json
 * @param param
 * @return dag::Graph*
 */
dag::Graph *createStableDiffusionText2ImageGraph(
    const std::string &name, dag::Edge *prompt, dag::Edge *negative_prompt,
    dag::Edge *output, base::InferenceType clip_inference_type,
    SchedulerType scheduler_type, base::InferenceType unet_inference_type,
    base::InferenceType vae_inference_type, std::vector<base::Param *> &param) {
  // graph
  dag::Graph *graph = new dag::Graph(name, {prompt, negative_prompt}, {output});

  // clip
  dag::Edge *text_embeddings = graph->createEdge("text_embeddings");
  dag::Graph *clip =
      createCLIPGraph("clip", prompt, negative_prompt, text_embeddings,
                      clip_inference_type, param);
  graph->addNode(clip, false);

  // scheduler_unet
  dag::Edge *latent = graph->createEdge("latent");
  std::vector<base::Param *>::iterator it = param.begin() + 4;
  std::vector<base::Param *> schedule_unet_param(it, it + 2);
  dag::Graph *text2images_scheduler_unet = createText2ImagesSchedulerUNetGraph(
      "scheduler_unet", text_embeddings, latent, scheduler_type,
      unet_inference_type, schedule_unet_param);
  graph->addNode(text2images_scheduler_unet, false);

  // vae
  dag::Node *vae =
      graph->createInfer<Infer>("vae", vae_inference_type, latent, output);
  vae->setParam(param[5]);

  return graph;
}

}  // namespace model
}  // namespace nndeploy