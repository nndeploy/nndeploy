#include "nndeploy/stable_diffusion/denoise.h"

#include "nndeploy/infer/infer.h"
#include "nndeploy/op/op_add.h"
#include "nndeploy/op/op_concat.h"
#include "nndeploy/op/op_muls.h"
#include "nndeploy/op/op_split.h"
#include "nndeploy/op/op_sub.h"
#include "nndeploy/stable_diffusion/ddim_scheduler.h"
#include "nndeploy/stable_diffusion/scheduler.h"
#include "nndeploy/stable_diffusion/utils.h"

namespace nndeploy {
namespace stable_diffusion {

class NNDEPLOY_CC_API DenoiseGraph : public dag::Graph {
 public:
  DenoiseGraph(const std::string name, SchedulerType scheduler_type,
               std::vector<dag::Edge *> inputs,
               std::vector<dag::Edge *> outputs)
      : dag::Graph(name, inputs, outputs), scheduler_type_(scheduler_type) {
    param_ = std::make_shared<DDIMSchedulerParam>();
    scheduler_ = createScheduler(scheduler_type_);
  }
  ~DenoiseGraph() {}

  base::Status init() {
    base::Status status = base::kStatusCodeOk;
    status = this->construct();
    NNDEPLOY_RETURN_ON_NEQ(status, base::kStatusCodeOk,
                           "denoise graph construct failed.");

    status = this->executor();
    NNDEPLOY_RETURN_ON_NEQ(status, base::kStatusCodeOk,
                           "denoise graph executor failed!");

    scheduler_param_ = (SchedulerParam *)(param_.get());
    status = scheduler_->init(scheduler_param_);
    NNDEPLOY_RETURN_ON_NEQ(status, base::kStatusCodeOk,
                           "scheduler init failed!");
    setInitializedFlag(true);

    return status;
  }

  base::Status deinit() {
    base::Status status = base::kStatusCodeOk;
    status = executor_->deinit();
    NNDEPLOY_RETURN_ON_NEQ(status, base::kStatusCodeOk,
                           "denoise graph executor deinit failed!");
    setInitializedFlag(false);
    return status;
  }

  base::Status run() {
    base::Status status = base::kStatusCodeOk;
    setRunningFlag(true);

    dag::Edge *prev_latents = this->getEdge("prev_latents");
    device::Device *device = device::getDefaultHostDevice();
    device::TensorDesc latents_desc;
    latents_desc.data_type_ = base::dataTypeOf<float>();
    latents_desc.data_format_ = base::kDataFormatNCHW;
    latents_desc.shape_ = {1, 4, 64, 64};
    device::Tensor *prev_latents_t =
        prev_latents->create(device, latents_desc, 0);
    // prev_latents_t->set(1.0f);
    std::mt19937 generator;
    initializeLatents(generator, init_noise_sigma_, prev_latents_t);

    dag::Edge *timestep = this->getEdge("denoise_timestep");
    device::TensorDesc timestep_desc;
    timestep_desc.data_type_ = base::dataTypeOf<float>();
    timestep_desc.data_format_ = base::kDataFormatNC;
    timestep_desc.shape_ = {1};
    device::Tensor *timestep_t = timestep->create(device, timestep_desc, 0);

    scheduler_->setTimesteps();
    std::vector<int> timesteps = scheduler_->getTimestep();
    ProgressBar progress(timesteps.size(), 80, "Progress", "Processing...");
    int i = 0;
    for (const auto &val : timesteps) {
      timestep_t->set((float)val);
      status = executor_->run();
      NNDEPLOY_RETURN_ON_NEQ(status, base::kStatusCodeOk,
                             "executor run failed!");

      device::Tensor *latents = this->getOutput(0)->getTensor(this);
      latents->copyTo(prev_latents_t);
      progress.update(i++);
    }
    progress.finish();
    setRunningFlag(false);
    return status;
  }

  float getTimestep() { return timestep_; }

  Scheduler *getScheduler() { return scheduler_; }

 private:
  SchedulerType scheduler_type_ = kSchedulerTypeNotSupport;
  Scheduler *scheduler_ = nullptr;
  SchedulerParam *scheduler_param_ = nullptr;

  float timestep_ = 0.0;
  float init_noise_sigma_ = 1.0f;
};

class NNDEPLOY_CC_API InitLatentsNode : public dag::Node {
 public:
  InitLatentsNode(const std::string &name, std::vector<dag::Edge *> inputs,
                  std::vector<dag::Edge *> outputs)
      : dag::Node(name, inputs, outputs) {}

  virtual ~InitLatentsNode() {}

  virtual base::Status init() {
    device::Device *device = device::getDefaultHostDevice();
    // device::TensorDesc latent_desc;
    // latent_desc.data_type_ = base::dataTypeOf<float>();
    // latent_desc.data_format_ = base::kDataFormatNCHW;
    // latent_desc.shape_ = {1, 4, 64, 64};
    // latent = this->getOutput(1)->create(device, latent_desc, 0);

    // std::mt19937 generator;
    // initializeLatents(generator, init_noise_sigma_, latent);

    do_classifier_free_guidance_ = (guidance_ > 1.0) ? true : false;
    if (do_classifier_free_guidance_) {
      device::TensorDesc latent_desc;
      latent_desc.data_type_ = base::dataTypeOf<float>();
      latent_desc.data_format_ = base::kDataFormatNCHW;
      latent_desc.shape_ = {2, 4, 64, 64};
      latents = this->getOutput(0)->create(device, latent_desc, 0);
    } else {
      device::TensorDesc latent_desc;
      latent_desc.data_type_ = base::dataTypeOf<float>();
      latent_desc.data_format_ = base::kDataFormatNCHW;
      latent_desc.shape_ = {1, 4, 64, 64};
      latents = this->getOutput(0)->create(device, latent_desc, 0);
    }

    // device::TensorDesc scalar_desc;
    // scalar_desc.data_type_ = base::dataTypeOf<float>();
    // scalar_desc.data_format_ = base::kDataFormatNC;
    // scalar_desc.shape_ = {1};
    // timestep_t = this->getOutput(2)->create(device, scalar_desc, 0);
    return base::kStatusCodeOk;
  }

  virtual base::Status deinit() {
    /** todo **/
    return base::kStatusCodeOk;
  }

  virtual base::Status run() {
    device::Tensor *latent = this->getInput(0)->getTensor(this);
    if (do_classifier_free_guidance_) {
      std::shared_ptr<ir::ConcatParam> param =
          std::make_shared<ir::ConcatParam>();
      param->axis_ = 0;
      op::concat({latent, latent}, param, latents);
    } else {
      latent->copyTo(latents);
    }
    // DenoiseGraph *graph = (DenoiseGraph *)(this->getGraph());
    // float timestep = graph->getTimestep();
    // timestep_t->set(timestep);

    return base::kStatusCodeOk;
  }

 private:
  float guidance_ = 7.5f;
  bool do_classifier_free_guidance_ = false;
  // float init_noise_sigma_ = 1.0f;
  device::Tensor *latents = nullptr;
};

class NNDEPLOY_CC_API DDIMScheduleNode : public dag::Node {
 public:
  DDIMScheduleNode(const std::string &name, std::vector<dag::Edge *> inputs,
                   std::vector<dag::Edge *> outputs)
      : dag::Node(name, inputs, outputs) {}

  virtual ~DDIMScheduleNode() {}

  virtual base::Status init() {
    base::Status status = base::kStatusCodeOk;
    DenoiseGraph *denoise_graph = (DenoiseGraph *)(this->getGraph());
    scheduler_ = denoise_graph->getScheduler();

    device::Device *device = device::getDefaultHostDevice();
    device::TensorDesc latent_desc;
    latent_desc.data_type_ = base::dataTypeOf<float>();
    latent_desc.data_format_ = base::kDataFormatNCHW;
    latent_desc.shape_ = {1, 4, 64, 64};
    latents = this->getOutput(0)->create(device, latent_desc, 0);

    device::TensorDesc noise_desc;
    noise_desc.data_type_ = base::dataTypeOf<float>();
    noise_desc.data_format_ = base::kDataFormatNCHW;
    noise_desc.shape_ = {1, 4, 64, 64};
    noise_pred_ = new device::Tensor(device, noise_desc);
    noise_pred_uncond_ = new device::Tensor(device, noise_desc);
    noise_sub_ = new device::Tensor(device, noise_desc);
    noise_muls_ = new device::Tensor(device, noise_desc);

    device::TensorDesc scalar_desc;
    scalar_desc.data_type_ = base::dataTypeOf<float>();
    scalar_desc.data_format_ = base::kDataFormatNC;
    scalar_desc.shape_.emplace_back(1);
    scalar_ = new device::Tensor(device, scalar_desc);
    scalar_->set(guidance_);

    do_classifier_free_guidance_ = (guidance_ > 1.0) ? true : false;

    return status;
  }

  virtual base::Status deinit() { return base::kStatusCodeOk; }

  virtual base::Status run() {
    device::Tensor *unet_output_t = this->getInput(0)->getTensor(this);
    if (do_classifier_free_guidance_) {
      std::vector<device::Tensor *> outputs = {noise_pred_uncond_, noise_pred_};
      std::shared_ptr<ir::SplitParam> split_param =
          std::make_shared<ir::SplitParam>();
      split_param->axis_ = 0;
      split_param->num_outputs_ = 2;
      op::split(unet_output_t, split_param, outputs);
      op::sub(noise_pred_, noise_pred_uncond_, noise_sub_);
      op::muls(scalar_, noise_sub_, noise_muls_);
      op::add(noise_pred_uncond_, noise_muls_, noise_pred_);
    } else {
      noise_pred_ = unet_output_t;
    }

    device::Tensor *prev_latents = this->getInput(1)->getTensor(this);
    device::Tensor *timestep_t = this->getInput(2)->getTensor(this);
    scheduler_->step(noise_pred_, timestep_t, prev_latents, latents);

    return base::kStatusCodeOk;
  }

 private:
  Scheduler *scheduler_ = nullptr;
  SchedulerParam *scheduler_param_ = nullptr;

  float guidance_ = 7.5f;
  bool do_classifier_free_guidance_ = false;

  // intermediate tensors
  device::Tensor *latent = nullptr;
  device::Tensor *latents = nullptr;
  device::Tensor *noise_pred_ = nullptr;
  device::Tensor *noise_pred_uncond_ = nullptr;
  device::Tensor *noise_sub_ = nullptr;
  device::Tensor *noise_muls_ = nullptr;
  device::Tensor *scalar_ = nullptr;
  device::Tensor *timestep_t = nullptr;
};

dag::Graph *createDenoiseGraph(const std::string &name,
                               dag::Edge *text_embeddings, dag::Edge *latents,
                               SchedulerType scheduler_type,
                               base::InferenceType inference_type,
                               std::vector<base::Param *> &param) {
  dag::Graph *denoise_graph =
      new DenoiseGraph(name, scheduler_type, {text_embeddings}, {latents});

  dag::Edge *prev_latents = denoise_graph->createEdge("prev_latents");
  dag::Edge *timestep = denoise_graph->createEdge("denoise_timestep");
  dag::Edge *model_input = denoise_graph->createEdge("model_input");
  dag::Node *init_latents_node = denoise_graph->createNode<InitLatentsNode>(
      "init_latents",
      std::vector<dag::Edge *>{prev_latents, text_embeddings, timestep},
      std::vector<dag::Edge *>{model_input});
  init_latents_node->setGraph(denoise_graph);

  dag::Edge *model_output = denoise_graph->createEdge("unet_output");
  dag::Node *unet_node = denoise_graph->createInfer<infer::Infer>(
      "unet", inference_type, {text_embeddings, model_input, timestep},
      {model_output});
  inference::InferenceParam *infer_param = new inference::InferenceParam();
  infer_param->device_type_ = base::kDeviceTypeCodeCuda;
  infer_param->model_type_ = base::kModelTypeOnnx;
  infer_param->is_path_ = true;
  std::vector<std::string> onnx_path = {
      "/home/lds/stable-diffusion.onnx/models/unet/model.onnx"};
  infer_param->model_value_ = onnx_path;
  unet_node->setParam(infer_param);
  //   base::DeviceType device_type(base::kDeviceTypeCodeCuda, 0);
  //   unet_node->setDeviceType(device_type);

  DDIMScheduleNode *ddim_schedule_node =
      (DDIMScheduleNode *)(denoise_graph->createNode<DDIMScheduleNode>(
          "ddim_schedule", {model_output, prev_latents, timestep}, {latents}));
  ddim_schedule_node->setGraph(denoise_graph);

  return denoise_graph;
}
}  // namespace stable_diffusion
}  // namespace nndeploy