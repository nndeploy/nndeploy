#include "nndeploy/stable_diffusion/clip.h"

#include "nndeploy/infer/infer.h"
#include "nndeploy/op/op_concat.h"
#include "nndeploy/stable_diffusion/ddim_scheduler.h"
#include "nndeploy/stable_diffusion/utils.h"
#include "nndeploy/tokenizer/tokenizer_cpp/tokenizer_cpp.h"

namespace nndeploy {
namespace stable_diffusion {

class NNDEPLOY_CC_API CvtTokenIds2Tensor : public dag::Node {
 public:
  CvtTokenIds2Tensor(const std::string &name, std::vector<dag::Edge *> inputs,
                     std::vector<dag::Edge *> outputs)
      : dag::Node(name, inputs, outputs) {
    key_ = "nndeploy::stable_diffusion::CvtTokenIds2Tensor";
    desc_ =
        "Convert TokenizerIds to int32 NC tensor with BOS=49406 and PAD=49407.";
    this->setInputTypeInfo<tokenizer::TokenizerIds>();
    this->setOutputTypeInfo<device::Tensor>();
  }

  virtual ~CvtTokenIds2Tensor() {}

  virtual base::Status run() {
    tokenizer::TokenizerIds *input =
        (tokenizer::TokenizerIds *)(this->getInput(0)->getParam(this));
    std::vector<std::vector<int32_t>> ids = input->ids_;

    device::Device *device = device::getDefaultHostDevice();
    if (device == nullptr) {
      NNDEPLOY_LOGE("device is nullptr\n");
      return base::kStatusCodeErrorInvalidParam;
    }
    device::TensorDesc desc(base::dataTypeOf<int32_t>(), base::kDataFormatNC,
                            {1, max_length_});
    device::Tensor *output = this->getOutput(0)->create(device, desc);
    output->set(49407);

    int32_t *value = (int32_t *)output->getData();
    value[0] = 49406;
    for (int i = 0; i < ids[0].size(); i++) {
      value[i + 1] = ids[0][i];
    }
    this->getOutput(0)->notifyWritten(output);

    return base::kStatusCodeOk;
  }

 private:
  int32_t max_length_ = 77;
};

class NNDEPLOY_CC_API ConcatEmbedding : public dag::Node {
 public:
  ConcatEmbedding(const std::string &name, std::vector<dag::Edge *> inputs,
                  std::vector<dag::Edge *> outputs)
      : dag::Node(name, inputs, outputs) {
    key_ = "nndeploy::stable_diffusion::ConcatEmbedding";
    desc_ =
        "Concatenate prompt and negative prompt embeddings for classifier-free "
        "guidance.";
    this->setInputTypeInfo<device::Tensor>();
    this->setInputTypeInfo<device::Tensor>();
    this->setOutputTypeInfo<device::Tensor>();
  }

  virtual ~ConcatEmbedding() {};

  base::Status setGuidance(float guidance) {
    guidance_ = guidance;
    return base::kStatusCodeOk;
  }

  virtual base::Status run() {
    bool do_classifier_free_guidance = (guidance_ > 1.0) ? true : false;

    device::Tensor *prompt = this->getInput(0)->getTensor(this);
    device::Tensor *negative_prompt = this->getInput(1)->getTensor(this);

    device::Device *device = device::getDevice(device_type_);
    if (device == nullptr) {
      NNDEPLOY_LOGE("device is nullptr\n");
      return base::kStatusCodeErrorInvalidParam;
    }
    std::vector<int> shape = prompt->getShape();

    device::Tensor *output = nullptr;
    if (do_classifier_free_guidance) {
      shape[0] = shape[0] * 2;
      device::TensorDesc desc(base::dataTypeOf<float>(), base::kDataFormatNCL,
                              shape);
      output = this->getOutput(0)->create(device, desc);

      std::shared_ptr<ir::ConcatParam> param =
          std::make_shared<ir::ConcatParam>();
      param->axis_ = 0;
      op::concat({negative_prompt, prompt}, param, output);
    } else {
      device::TensorDesc desc(base::dataTypeOf<float>(), base::kDataFormatNCL,
                              shape);
      output = this->getOutput(0)->create(device, desc);
      prompt->copyTo(output);
    }
    this->getOutput(0)->notifyWritten(output);

    return base::kStatusCodeOk;
  }

  virtual base::Status serialize(
      rapidjson::Value &json, rapidjson::Document::AllocatorType &allocator) {
    base::Status status = dag::Node::serialize(json, allocator);
    if (status != base::kStatusCodeOk) {
      return status;
    }
    json.AddMember("guidance_", guidance_, allocator);
    return status;
  }

  virtual base::Status deserialize(rapidjson::Value &json) {
    base::Status status = dag::Node::deserialize(json);
    if (status != base::kStatusCodeOk) {
      return status;
    }
    if (json.HasMember("guidance_") && json["guidance_"].IsFloat()) {
      int guidance = json["guidance_"].GetFloat();
      this->setGuidance(guidance);
    }
    return status;
  }

 private:
  float guidance_ = 7.5;
};

class NNDEPLOY_CC_API EmbeddingGraph : public dag::Graph {
 public:
  EmbeddingGraph(const std::string &name, std::vector<dag::Edge *> inputs,
                 std::vector<dag::Edge *> outputs)
      : dag::Graph(name, inputs, outputs) {
    key_ = "nndeploy::stable_diffusion::EmbeddingGraph";
    desc_ =
        "Stable Diffusion text-to-embedding graph: TokenizerText -> "
        "TokenizerEncodeCpp -> CvtTokenIds2Tensor -> CLIP inference -> Tensor.";
    this->setInputTypeInfo<tokenizer::TokenizerText>();
    this->setOutputTypeInfo<device::Tensor>();
    tokenize_ = dynamic_cast<tokenizer::TokenizerEncodeCpp *>(
        this->createNode<tokenizer::TokenizerEncodeCpp>("tokenize"));
    cvt_ = dynamic_cast<CvtTokenIds2Tensor *>(
        this->createNode<CvtTokenIds2Tensor>("cvt_token"));
    clip_infer_ =
        dynamic_cast<infer::Infer *>(this->createNode<infer::Infer>("infer"));
  };

  virtual ~EmbeddingGraph() {}

  virtual base::Status setNodeExternalParam() {
    tokenizer::TokenizerPraram *tokenize_param =
        dynamic_cast<tokenizer::TokenizerPraram *>(tokenize_->getParam());
    tokenizer::TokenizerPraram *tokenize_param_ =
        dynamic_cast<tokenizer::TokenizerPraram *>(
            getExternalParam("tokenize_param").get());
    *tokenize_param = *tokenize_param_;

    inference::InferenceParam *infer_param =
        dynamic_cast<inference::InferenceParam *>(clip_infer_->getParam());
    inference::InferenceParam *infer_param_ =
        dynamic_cast<inference::InferenceParam *>(
            getExternalParam("clip_infer_param").get());
    *infer_param = *infer_param_;
    return base::kStatusCodeOk;
  }

  base::Status make(const dag::NodeDesc &tokenize_desc,
                    const dag::NodeDesc &cvt_desc,
                    const dag::NodeDesc &infer_desc,
                    base::InferenceType inference_type) {
    base::Status status = base::kStatusCodeOk;
    this->setNodeDesc(tokenize_, tokenize_desc);
    this->setNodeDesc(cvt_, cvt_desc);
    this->setNodeDesc(clip_infer_, infer_desc);
    status = clip_infer_->setInferenceType(inference_type);
    if (status != base::kStatusCodeOk) {
      NNDEPLOY_LOGE("Failed to set inference type");
      return status;
    }
    this->setNodeExternalParam();
    return status;
  }

 private:
  dag::Node *tokenize_;
  dag::Node *cvt_;
  infer::Infer *clip_infer_;
};

class NNDEPLOY_CC_API ClipGraph : public dag::Graph {
 public:
  ClipGraph(const std::string &name) : dag::Graph(name) {
    key_ = "nndeploy::stable_diffusion::ClipGraph";
    desc_ =
        "CLIP text encoder graph: [prompt, negative_prompt] -> embeddings -> "
        "concatenated text_embedding.";
    this->setInputTypeInfo<tokenizer::TokenizerText>();
    this->setInputTypeInfo<tokenizer::TokenizerText>();
    this->setOutputTypeInfo<device::Tensor>();
    embedding_ = dynamic_cast<EmbeddingGraph *>(
        this->createNode<EmbeddingGraph>("embeddings"));
    negative_embedding_ = dynamic_cast<EmbeddingGraph *>(
        this->createNode<EmbeddingGraph>("negative_embeddings"));
    concat_ = dynamic_cast<ConcatEmbedding *>(
        this->createNode<ConcatEmbedding>("concat"));
  }

  ClipGraph(const std::string &name, std::vector<dag::Edge *> inputs,
            std::vector<dag::Edge *> outputs)
      : dag::Graph(name, inputs, outputs) {
    key_ = "nndeploy::stable_diffusion::ClipGraph";
    desc_ =
        "CLIP text encoder graph: [prompt, negative_prompt] -> embeddings -> "
        "concatenated text_embedding.";
    this->setInputTypeInfo<tokenizer::TokenizerText>();
    this->setInputTypeInfo<tokenizer::TokenizerText>();
    this->setOutputTypeInfo<device::Tensor>();
    embedding_ = dynamic_cast<EmbeddingGraph *>(
        this->createNode<EmbeddingGraph>("embeddings"));
    negative_embedding_ = dynamic_cast<EmbeddingGraph *>(
        this->createNode<EmbeddingGraph>("negative_embeddings"));
    concat_ = dynamic_cast<ConcatEmbedding *>(
        this->createNode<ConcatEmbedding>("concat"));
  }

  virtual ~ClipGraph() {}

  base::Status make(const dag::NodeDesc &embedding_desc,
                    const dag::NodeDesc &negative_embedding_desc,
                    const dag::NodeDesc &concat_desc,
                    base::InferenceType inference_type) {
    this->setNodeDesc(embedding_, embedding_desc);
    this->setNodeDesc(negative_embedding_, negative_embedding_desc);
    this->setNodeDesc(concat_, concat_desc);

    dag::NodeDesc tokenize_desc("tokenize", embedding_desc.getInputs(),
                                {"token_ids"});
    dag::NodeDesc cvt_desc("cvt", {"token_ids"}, {"infer_ids"});
    dag::NodeDesc infer_desc("clip_infer", {"infer_ids"}, {"prompt_ids"});
    embedding_->make(tokenize_desc, cvt_desc, infer_desc, inference_type);

    dag::NodeDesc ne_tokenize_desc("negative_tokenize",
                                   negative_embedding_desc.getInputs(),
                                   {"token_ids"});
    dag::NodeDesc ne_cvt_desc("negative_cvt", {"token_ids"}, {"infer_ids"});
    dag::NodeDesc ne_infer_desc("negative_clip_infer", {"infer_ids"},
                                {"negative_prompt_ids"});
    negative_embedding_->make(ne_tokenize_desc, ne_cvt_desc, ne_infer_desc,
                              inference_type);
    return base::kStatusCodeOk;
  }

 private:
  EmbeddingGraph *embedding_ = nullptr;
  EmbeddingGraph *negative_embedding_ = nullptr;
  dag::Node *concat_ = nullptr;
};

dag::Graph *createCLIPGraph(const std::string &name, dag::Edge *prompt,
                            dag::Edge *negative_prompt, dag::Edge *output,
                            base::InferenceType inference_type,
                            std::vector<base::Param *> &param) {
  Text2ImageParam *text2image_param = (Text2ImageParam *)param[0];
  auto tokenizer_param = std::make_shared<tokenizer::TokenizerPraram>();
  tokenizer_param->tokenizer_type_ = tokenizer::TokenizerType::kTokenizerTypeHF;
  tokenizer_param->is_path_ = true;
  tokenizer_param->json_blob_ = text2image_param->model_value_[0];

  auto infer_param = std::make_shared<inference::InferenceParam>();
  infer_param->device_type_ = text2image_param->device_type_;
  infer_param->model_type_ = text2image_param->model_type_;
  infer_param->is_path_ = text2image_param->is_path_;
  std::vector<std::string> onnx_path = {text2image_param->model_value_[1]};
  infer_param->model_value_ = onnx_path;

  ClipGraph *clip_graph =
      new ClipGraph(name, {prompt, negative_prompt}, {output});
  clip_graph->setExternalParam("tokenize_param", tokenizer_param);
  clip_graph->setExternalParam("clip_infer_param", infer_param);
  dag::NodeDesc embedding_desc("embedding", {prompt->getName()},
                               {"prompt_ids"});
  dag::NodeDesc ne_embedding_desc("ne_embedding", {negative_prompt->getName()},
                                  {"negative_prompt_ids"});
  dag::NodeDesc concat_desc("concat", {"prompt_ids", "negative_prompt_ids"},
                            {output->getName()});
  clip_graph->make(embedding_desc, ne_embedding_desc, concat_desc,
                   inference_type);
  return clip_graph;
}

REGISTER_NODE("nndeploy::stable_diffusion::CvtTokenIds2Tensor",
              CvtTokenIds2Tensor);
REGISTER_NODE("nndeploy::stable_diffusion::ConcatEmbedding", ConcatEmbedding);
REGISTER_NODE("nndeploy::stable_diffusion::EmbeddingGraph", EmbeddingGraph);
REGISTER_NODE("nndeploy::stable_diffusion::ClipGraph", ClipGraph);

}  // namespace stable_diffusion
}  // namespace nndeploy