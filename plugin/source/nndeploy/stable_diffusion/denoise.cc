#include "nndeploy/stable_diffusion/denoise.h"

#include "nndeploy/dag/composite_node.h"
#include "nndeploy/dag/edge/pipeline_edge.h"
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

class NNDEPLOY_CC_API InitLatents : public dag::Node {
 public:
  InitLatents(const std::string &name, std::vector<dag::Edge *> inputs,
              std::vector<dag::Edge *> outputs)
      : dag::Node(name, inputs, outputs) {
    key_ = "nndeploy::stable_diffusion::InitLatents";
    desc_ = "init latents [latent image]";
    param_ = std::make_shared<DDIMSchedulerParam>();
    this->setOutputTypeInfo<device::Tensor>();
    node_type_ = dag::NodeType::kNodeTypeInput;
  }

  virtual ~InitLatents() {}

  virtual base::Status run() {
    std::shared_ptr<DDIMSchedulerParam> scheduler_param =
        std::dynamic_pointer_cast<DDIMSchedulerParam>(param_);
    device::Device *device = device::getDefaultHostDevice();
    device::TensorDesc latents_desc;
    latents_desc.data_type_ = base::dataTypeOf<float>();
    latents_desc.data_format_ = base::kDataFormatNCHW;
    latents_desc.shape_ = {1, scheduler_param->unet_channels_,
                           scheduler_param->image_height_ / 8,
                           scheduler_param->image_width_ / 8};
    device::Tensor *latents = this->getOutput(0)->create(device, latents_desc);

    std::mt19937 generator;
    initializeLatents(generator, scheduler_param->init_noise_sigma_, latents);

    this->getOutput(0)->notifyWritten(latents);

    index_++;

    return base::kStatusCodeOk;
  }

  virtual base::EdgeUpdateFlag updateInput() {
    if (index_ < size_) {
      return base::kEdgeUpdateFlagComplete;
    } else {
      if (size_ == 0) {
        return base::kEdgeUpdateFlagComplete;
      } else {
        return base::kEdgeUpdateFlagTerminate;
      }
    }
  }

  void setSize(int size) {
    if (size > 0) {
      size_ = size;
    }
  }
  int getSize() { return size_; }

  virtual base::Status serialize(
      rapidjson::Value &json, rapidjson::Document::AllocatorType &allocator) {
    base::Status status = dag::Node::serialize(json, allocator);
    if (status != base::kStatusCodeOk) {
      return status;
    }
    json.AddMember("size_", size_, allocator);
    return status;
  }

  virtual base::Status deserialize(rapidjson::Value &json) {
    base::Status status = dag::Node::deserialize(json);
    if (status != base::kStatusCodeOk) {
      return status;
    }
    if (json.HasMember("size_") && json["size_"].IsInt()) {
      int size = json["size_"].GetInt();
      if (size > 0) {
        this->setSize(size);
      }
    }
    return status;
  }

 private:
  int index_ = 0;
  int size_ = 1;
};

class NNDEPLOY_CC_API DDIMSchedule : public dag::Node {
 public:
  DDIMSchedule(const std::string &name, std::vector<dag::Edge *> inputs,
               std::vector<dag::Edge *> outputs)
      : dag::Node(name, inputs, outputs) {
    key_ = "nndeploy::stable_diffusion::DDIMSchedule";
    desc_ = "ddim schedule [unet_output/latents/timestep -> latents]";
    param_ = std::make_shared<DDIMSchedulerParam>();
    this->setInputTypeInfo<device::Tensor>();
    this->setInputTypeInfo<device::Tensor>();
    this->setInputTypeInfo<device::Tensor>();
    this->setOutputTypeInfo<device::Tensor>();
  }

  virtual ~DDIMSchedule() {}

  void setScheduler(Scheduler *scheduler) { scheduler_ = scheduler; }

  virtual base::Status init() {
    base::Status status = base::kStatusCodeOk;
    DDIMSchedulerParam *ddim_param = (DDIMSchedulerParam *)param_.get();

    device::Device *device = device::getDefaultHostDevice();
    device::TensorDesc noise_desc;
    noise_desc.data_type_ = base::dataTypeOf<float>();
    noise_desc.data_format_ = base::kDataFormatNCHW;
    noise_desc.shape_ = {1, ddim_param->unet_channels_,
                         ddim_param->image_height_ / 8,
                         ddim_param->image_width_ / 8};
    noise_pred_ = new device::Tensor(device, noise_desc);
    noise_pred_uncond_ = new device::Tensor(device, noise_desc);
    noise_sub_ = new device::Tensor(device, noise_desc);
    noise_muls_ = new device::Tensor(device, noise_desc);

    device::TensorDesc scalar_desc;
    scalar_desc.data_type_ = base::dataTypeOf<float>();
    scalar_desc.data_format_ = base::kDataFormatNC;
    scalar_desc.shape_.emplace_back(1);
    scalar_ = new device::Tensor(device, scalar_desc);
    scalar_->set(ddim_param->guidance_scale_);

    do_classifier_free_guidance_ =
        (ddim_param->guidance_scale_ > 1.0) ? true : false;

    return status;
  }

  virtual base::Status deinit() {
    if (noise_pred_ != nullptr) {
      delete noise_pred_;
      noise_pred_ = nullptr;
    }
    if (noise_pred_uncond_ != nullptr) {
      delete noise_pred_uncond_;
      noise_pred_uncond_ = nullptr;
    }
    if (noise_sub_ != nullptr) {
      delete noise_sub_;
      noise_sub_ = nullptr;
    }
    if (noise_muls_ != nullptr) {
      delete noise_muls_;
      noise_muls_ = nullptr;
    }
    if (scalar_ != nullptr) {
      delete scalar_;
      scalar_ = nullptr;
    }
    return base::kStatusCodeOk;
  }

  virtual base::Status run() {
    DDIMSchedulerParam *ddim_param = (DDIMSchedulerParam *)param_.get();
    device::Device *device = device::getDefaultHostDevice();
    device::TensorDesc latent_desc;
    latent_desc.data_type_ = base::dataTypeOf<float>();
    latent_desc.data_format_ = base::kDataFormatNCHW;
    latent_desc.shape_ = {1, ddim_param->unet_channels_,
                          ddim_param->image_height_ / 8,
                          ddim_param->image_width_ / 8};
    latents = this->getOutput(0)->create(device, latent_desc);

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

    // this->getOutput(0)->notifyWritten(latents);

    return base::kStatusCodeOk;
  }

 private:
  Scheduler *scheduler_ = nullptr;

  bool do_classifier_free_guidance_ = false;

  device::Tensor *latent = nullptr;
  device::Tensor *latents = nullptr;
  device::Tensor *noise_pred_ = nullptr;
  device::Tensor *noise_pred_uncond_ = nullptr;
  device::Tensor *noise_sub_ = nullptr;
  device::Tensor *noise_muls_ = nullptr;
  device::Tensor *scalar_ = nullptr;
  device::Tensor *timestep_t = nullptr;
};

class NNDEPLOY_CC_API Denoise : public dag::CompositeNode {
 public:
  Denoise(const std::string &name, std::vector<dag::Edge *> inputs,
          std::vector<dag::Edge *> outputs)
      : CompositeNode(name, inputs, outputs),
        scheduler_type_(stable_diffusion::kSchedulerTypeDDIM) {
    key_ = "nndeploy::stable_diffusion::Denoise";
    desc_ = "denoise latents composite node[cfg->unet->ddim]";
    this->setInputTypeInfo<device::Tensor>();
    this->setInputTypeInfo<device::Tensor>();
    this->setOutputTypeInfo<device::Tensor>();

    infer_ = (infer::Infer *)this->createNode<infer::Infer>("unet_infer");
    ddim_schedule_ =
        (DDIMSchedule *)this->createNode<DDIMSchedule>("ddim_schedule");

    // dag::NodeDesc infer_desc("unet_infer",
    //                          {"embeddings", "cfg_latents", "timestep"},
    //                          {"unet_output"});
    // this->setNodeDesc(infer_, infer_desc);

    // dag::NodeDesc schedule_desc("ddim_schedule",
    //                             {"unet_output", "prev_latents", "timestep"},
    //                             {"latents"});
    // this->setNodeDesc(ddim_schedule_, schedule_desc);

    scheduler_ = createScheduler(scheduler_type_);
  }

  virtual ~Denoise() {
    if (scheduler_ != nullptr) {
      delete scheduler_;
      scheduler_ = nullptr;
    }
  }

  virtual base::Status init() {
    base::Status status = base::kStatusCodeOk;

    device_ = device::getDefaultHostDevice();

    auto schedule_param_ =
        dynamic_cast<DDIMSchedulerParam *>(ddim_schedule_->getParam());

    status = scheduler_->init(schedule_param_);
    NNDEPLOY_RETURN_ON_NEQ(status, base::kStatusCodeOk,
                           "scheduler init failed!");

    do_classifier_free_guidance_ =
        (schedule_param_->guidance_scale_ > 1.0) ? true : false;

    int chunk_num = do_classifier_free_guidance_ ? 2 : 1;

    device::TensorDesc latents_desc;
    latents_desc.data_type_ = base::dataTypeOf<float>();
    latents_desc.data_format_ = base::kDataFormatNCHW;
    latents_desc.shape_ = {chunk_num, schedule_param_->unet_channels_,
                           schedule_param_->image_height_ / 8,
                           schedule_param_->image_width_ / 8};
    dag::Edge *cfg_latents_edge = this->getEdge("cfg_latents");
    cfg_latents_ = cfg_latents_edge->create(device_, latents_desc);

    device::TensorDesc timesteps_desc;
    timesteps_desc.data_type_ = base::dataTypeOf<float>();
    timesteps_desc.data_format_ = base::kDataFormatNC;
    timesteps_desc.shape_ = {1};
    dag::Edge *timestep_edge = this->getEdge("timestep");
    timestep_ = timestep_edge->create(device_, timesteps_desc);

    infer_->init();

    ddim_schedule_->setScheduler(scheduler_);
    ddim_schedule_->init();

    return status;
  }

  virtual base::Status deinit() {
    base::Status status = base::kStatusCodeOk;
    status = scheduler_->deinit();
    NNDEPLOY_RETURN_ON_NEQ(status, base::kStatusCodeOk,
                           "scheduler deinit failed!");
    status = infer_->deinit();
    NNDEPLOY_RETURN_ON_NEQ(status, base::kStatusCodeOk,
                           "infer node deinit failed!");
    status = ddim_schedule_->deinit();
    NNDEPLOY_RETURN_ON_NEQ(status, base::kStatusCodeOk,
                           "schedule node deinit failed!");
    return status;
  }

  virtual base::Status run() {
    base::Status status = base::kStatusCodeOk;

    auto schedule_param_ =
        dynamic_cast<DDIMSchedulerParam *>(ddim_schedule_->getParam());

    device::Tensor *init_latents = this->getInput(0)->getTensor(this);
    device::TensorDesc latents_desc;
    latents_desc.data_type_ = base::dataTypeOf<float>();
    latents_desc.data_format_ = base::kDataFormatNCHW;
    latents_desc.shape_ = {1, schedule_param_->unet_channels_,
                           schedule_param_->image_height_ / 8,
                           schedule_param_->image_width_ / 8};
    dag::Edge *prev_latents_edge = this->getEdge("prev_latents");
    device::Tensor *prev_latents =
        prev_latents_edge->create(device_, latents_desc);
    dag::Edge *latents_edge = this->getEdge("latents");
    device::Tensor *latents = latents_edge->create(device_, latents_desc);

    init_latents->copyTo(prev_latents);

    device::Tensor *text_embeddings = this->getInput(1)->getTensor(this);
    infer_->getInput(0)->set(text_embeddings, true);

    scheduler_->setTimesteps();
    std::vector<int> timesteps = scheduler_->getTimesteps();
    ProgressBar progress(timesteps.size(), 100, "Denoise", "Processing...");
    int i = 0;
    infer_->updateInput();
    for (const auto &val : timesteps) {
      if (do_classifier_free_guidance_) {
        std::shared_ptr<ir::ConcatParam> param =
            std::make_shared<ir::ConcatParam>();
        param->axis_ = 0;
        op::concat({prev_latents, prev_latents}, param, cfg_latents_);
      } else {
        prev_latents->copyTo(cfg_latents_);
      }
      timestep_->set((float)val);

      infer_->run();
      ddim_schedule_->run();
      latents->copyTo(prev_latents);
      progress.update(i++);
    }
    progress.finish();
    device::Tensor *latents_out =
        this->getOutput(0)->create(device_, latents_desc);
    latents->copyTo(latents_out);
    this->getOutput(0)->notifyWritten(latents_out);

    setRunningFlag(false);
    return status;
  }

  void setScheduleParam(DDIMSchedulerParam *ddim_param) {
    if (ddim_param == nullptr || ddim_schedule_ == nullptr) {
      return;
    }
    auto param = dynamic_cast<DDIMSchedulerParam *>(ddim_schedule_->getParam());
    if (param) {
      *param = *ddim_param;
    } else {
      ddim_schedule_->setParam(new DDIMSchedulerParam(*ddim_param));
    }
  }

  void setInferenceParam(inference::InferenceParam *infer_param) {
    if (infer_param == nullptr || infer_ == nullptr) {
      return;
    }
    auto param = dynamic_cast<inference::InferenceParam *>(infer_->getParam());
    if (param) {
      *param = *infer_param;
    } else {
      infer_->setParam(new inference::InferenceParam(*infer_param));
    }
  }

  void setInferenceType(base::InferenceType inference_type) {
    inference_type_ = inference_type;
    infer_->setInferenceType(inference_type_);
  }

 private:
  device::Device *device_ = nullptr;

  SchedulerType scheduler_type_;
  Scheduler *scheduler_ = nullptr;  // DDIMScheduler

  device::Tensor *cfg_latents_ = nullptr;
  device::Tensor *timestep_ = nullptr;

  infer::Infer *infer_ = nullptr;
  DDIMSchedule *ddim_schedule_ = nullptr;

  base::InferenceType inference_type_;

  bool do_classifier_free_guidance_ = false;
};

class NNDEPLOY_CC_API DenoiseGraph : public dag::Graph {
 public:
  DenoiseGraph(const std::string &name, std::vector<dag::Edge *> inputs,
               std::vector<dag::Edge *> outputs)
      : dag::Graph(name, inputs, outputs) {
    key_ = "nndeploy::stable_diffusion::DenoiseGraph";
    desc_ = "denoise graph [init_latents->denoise]";
    this->setInputTypeInfo<device::Tensor>();
    this->setOutputTypeInfo<device::Tensor>();
    init_latents_ = dynamic_cast<InitLatents *>(
        this->createNode<InitLatents>("init_latents"));
    denoise_ = dynamic_cast<Denoise *>(this->createNode<Denoise>("denoise"));
  }

  virtual ~DenoiseGraph() {}

  virtual base::Status defaultParam() {
    DDIMSchedulerParam *ddim_param_ = dynamic_cast<DDIMSchedulerParam *>(
        getExternalParam("schedule_param_").get());
    Text2ImageParam *text_image_param_ = dynamic_cast<Text2ImageParam *>(
        getExternalParam("text_image_param").get());

    inference::InferenceParam *infer_param = new inference::InferenceParam();
    infer_param->device_type_ = text_image_param_->device_type_;
    infer_param->model_type_ = text_image_param_->model_type_;
    infer_param->is_path_ = text_image_param_->is_path_;
    std::vector<std::string> onnx_path = {text_image_param_->model_value_[2]};
    infer_param->model_value_ = onnx_path;

    init_latents_->setParam(ddim_param_);
    denoise_->setScheduleParam(ddim_param_);
    denoise_->setInferenceParam(infer_param);

    return base::kStatusCodeOk;
  }

  base::Status make(const dag::NodeDesc init_latents_desc,
                    const dag::NodeDesc denoise_desc,
                    base::InferenceType inference_type) {
    this->setNodeDesc(init_latents_, init_latents_desc);
    this->setNodeDesc(denoise_, denoise_desc);
    denoise_->setInferenceType(inference_type);
    this->defaultParam();
    return base::kStatusCodeOk;
  }

 private:
  dag::Node *init_latents_ = nullptr;
  Denoise *denoise_ = nullptr;
};

dag::Graph *createDenoiseGraph(const std::string &name,
                               dag::Edge *text_embeddings, dag::Edge *latents,
                               SchedulerType scheduler_type,
                               base::InferenceType inference_type,
                               std::vector<base::Param *> &param, int iter) {
  auto text2image_param =
      std::make_shared<Text2ImageParam>(*((Text2ImageParam *)(param[0])));
  auto schedule_param_ =
      std::make_shared<DDIMSchedulerParam>(*((DDIMSchedulerParam *)(param[1])));

  DenoiseGraph *denoise_graph =
      new DenoiseGraph(name, {text_embeddings}, {latents});
  dag::NodeDesc init_latents_desc("init_latents", {}, {"init_latents"});
  dag::NodeDesc denoise_desc("denoise",
                             {"init_latents", text_embeddings->getName()},
                             {latents->getName()});
  denoise_graph->setExternalParam("schedule_param_", schedule_param_);
  denoise_graph->setExternalParam("text_image_param", text2image_param);
  denoise_graph->make(init_latents_desc, denoise_desc, inference_type);
  return denoise_graph;
}

REGISTER_NODE("nndeploy::stable_diffusion::InitLatents", InitLatents);
REGISTER_NODE("nndeploy::stable_diffusion::DDIMSchedule", DDIMSchedule);
REGISTER_NODE("nndeploy::stable_diffusion::Denoise", Denoise);
REGISTER_NODE("nndeploy::stable_diffusion::DenoiseGraph", DenoiseGraph);

}  // namespace stable_diffusion
}  // namespace nndeploy