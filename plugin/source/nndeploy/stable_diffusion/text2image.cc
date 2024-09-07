
#include "nndeploy/infer/infer.h"
#include "nndeploy/stable_diffusion/clip.h"
#include "nndeploy/stable_diffusion/scheduler.h"
#include "nndeploy/stable_diffusion/stable_diffusion.h"

// TODO: Edge要支持能够支持同名的情况
// TODO: Node的名字能够支持同名的情况

namespace nndeploy {
namespace stable_diffusion {

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

    status = scheduler_->init();
    NNDEPLOY_RETURN_ON_NEQ(status, base::kStatusCodeOk, "init failed!");

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

    status = scheduler_->deinit();
    NNDEPLOY_RETURN_ON_NEQ(status, base::kStatusCodeOk, "deinit failed!");

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
    dag ::Edge *sample = this->getEdge("sample");
    device::TensorDesc sample_desc;
    sample_desc.data_type_ = base::dataTypeOf<float>();
    sample_desc.data_format_ = base::kDataFormatNCHW;
    sample_desc.shape_.emplace_back(batch_size);
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
    timestep_desc.shape_.emplace_back(1);
    timestep_desc.shape_.emplace_back(1);
    device::Tensor *timestep_tensor =
        timestep->create(host_device, timestep_desc, index);

    // build latent
    device::TensorDesc latent_desc;
    latent_desc.data_type_ = base::dataTypeOf<float>();
    latent_desc.data_format_ = base::kDataFormatNCHW;
    latent_desc.shape_.emplace_back(batch_size / 2);
    latent_desc.shape_.emplace_back(scheduler_param->unet_channels_);
    latent_height = scheduler_param->image_height_ / 8;
    latent_desc.shape_.emplace_back(latent_height);
    latent_width = scheduler_param->image_width_ / 8;
    latent_desc.shape_.emplace_back(latent_width);
    device::Tensor *latent = outputs_[0]->create(host_device, latent_desc,
                                                 inputs_[0]->getIndex(this));
    std::mt19937 generator;
    initializeLatents(generator, init_noise_sigma_, latent);

    device::Tensor *noise_pred_uncond_ =
        new device::Tensor(host_device, latent_desc);  //
    device::Tensor *noise_pred_text_ =
        new device::Tensor(host_device, latent_desc);  //

    // for (int i = 0; i < size; i++) {
    //   // 算子
    //   op::concat({latent, latent}, 0, sample);
    //   status = executor_->run();
    //   NNDEPLOY_RETURN_ON_NEQ(status, base::kStatusCodeOk,
    //                          "executor runfailed!");
    //   op::split(sample, 0, noise_pred_uncond_, noise_pred_text_);
    //   scheduler_->step(noise_pred_uncond_, latent, i);
    // }

    // float scale = 1.0f / 0.18215f;
    // op::mul(latent, scale, latent);

    if (noise_pred_uncond_ != nullptr) {
      delete noise_pred_uncond_;
    }
    if (noise_pred_text_ != nullptr) {
      delete noise_pred_text_;
    }

    setRunningFlag(false);
    return status;
  }

 private:
  SchedulerType scheduler_type_ = kSchedulerTypeNotSupport;
  Scheduler *scheduler_ = nullptr;
  // standard deviation of the initial noise distribution
  device::Tensor *latent_model_input_ = nullptr;
  float init_noise_sigma_ = 1.0f;                // 初始噪声的标准差
  device::Tensor *noise_pred_uncond_ = nullptr;  //
  device::Tensor *noise_pred_text_ = nullptr;    //
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
  Text2ImagesSchedulerUNet *text2image_scheduler_unet =
      new Text2ImagesSchedulerUNet(name, scheduler_type, inputs, outputs);
  text2image_scheduler_unet->setParam(param[0]);
  // convert_to
  dag::Edge *encoder_hidden_states =
      text2image_scheduler_unet->createEdge("encoder_hidden_states");
  // dag::Node *tensor_convert_to =
  //     text2image_scheduler_unet->createInfer<TensorConvertTo>(
  //         "tensor_convert_to", inference_type, text_embeddings,
  //         encoder_hidden_states);
  // tensor_convert_to->setParam(param[1]);
  // text2image_scheduler_unet->addNode(tensor_convert_to, false);

  dag::Edge *sample = text2image_scheduler_unet->createEdge("sample");
  dag::Edge *timestep = text2image_scheduler_unet->createEdge("timestep");
  dag::Node *unet = text2image_scheduler_unet->createInfer<infer::Infer>(
      "unet", inference_type, {sample, timestep, encoder_hidden_states},
      outputs);
  unet->setParam(param[2]);
  text2image_scheduler_unet->addNode(unet, false);

  return text2image_scheduler_unet;
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
  std::vector<base::Param *> schedule_unet_param(it, it + 3);
  dag::Graph *text2image_scheduler_unet = createText2ImagesSchedulerUNetGraph(
      "scheduler_unet", text_embeddings, latent, scheduler_type,
      unet_inference_type, schedule_unet_param);
  graph->addNode(text2image_scheduler_unet, false);

  // vae
  dag::Node *vae = graph->createInfer<infer::Infer>("vae", vae_inference_type,
                                                    latent, output);
  vae->setParam(param[5]);

  return graph;
}

}  // namespace stable_diffusion
}  // namespace nndeploy