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
    desc_ = "TokenizerIds to device::Tensor";
    this->setInputTypeInfo<tokenizer::TokenizerIds>();
    this->setOutputTypeInfo<device::Tensor>();
  }

  virtual ~CvtTokenIds2Tensor() {}

  virtual base::Status run() {
    tokenizer::TokenizerIds *input =
        (tokenizer::TokenizerIds *)(this->getInput(0)->getParam(this));
    std::vector<std::vector<int32_t>> ids = input->ids_;

    device::Device *device = device::getDevice(device_type_);
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
    key_ = "nndeploy::stable_diffusion::Concat";
    desc_ = "concat embedding";
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
        "stable diffusion embedding "
        "graph[tokenIds->TokenizerEncodeCpp->CvtTokenIds2Tensor->infer->"
        "Tensor]";
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
    return status;
  }

  base::Status setTokenizerParam(tokenizer::TokenizerPraram *param) {
    tokenize_->setParam(param);
    return base::kStatusCodeOk;
  }

  base::Status setInferParam(inference::InferenceParam *param) {
    clip_infer_->setParam(param);
    return base::kStatusCodeOk;
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
    desc_ = "clip graph[[prompt, negative_prompt]->text_embedding]";
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
    desc_ = "clip graph[[prompt, negative_prompt]->text_embedding]";
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

  base::Status make(const dag::NodeDesc tokenize_desc,
                    const dag::NodeDesc negative_tokenize_desc,
                    const dag::NodeDesc &concat_desc,
                    base::InferenceType inference_type) {
    dag::NodeDesc cvt_desc("cvt_token", {"token_ids"}, {"infer_ids"});
    dag::NodeDesc infer_desc("clip_infer", {"infer_ids"}, {"prompt_ids"});
    embedding_->make(tokenize_desc, cvt_desc, infer_desc, inference_type);

    dag::NodeDesc negative_cvt_desc("negative_cvt_token", {"token_ids"},
                                    {"infer_ids"});
    dag::NodeDesc negative_infer_desc("negative_clip_infer", {"infer_ids"},
                                      {"negative_prompt_ids"});
    negative_embedding_->make(negative_tokenize_desc, negative_cvt_desc,
                              negative_infer_desc, inference_type);
    this->setNodeDesc(concat_, concat_desc);
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
  // get param
  Text2ImageParam *text2image_param = (Text2ImageParam *)param[0];
  tokenizer::TokenizerPraram *tokenizer_param =
      new tokenizer::TokenizerPraram();
  tokenizer_param->tokenizer_type_ = tokenizer::TokenizerType::kTokenizerTypeHF;
  tokenizer_param->is_path_ = true;
  // tokenizer_param->json_blob_ =
  //     "/home/lds/stable-diffusion.onnx/models/tokenizer/tokenizer.json";
  tokenizer_param->json_blob_ = text2image_param->model_value_[0];

  inference::InferenceParam *infer_param = new inference::InferenceParam();
  infer_param->device_type_ = text2image_param->device_type_;
  infer_param->model_type_ = text2image_param->model_type_;
  infer_param->is_path_ = text2image_param->is_path_;
  // std::vector<std::string> onnx_path = {
  //     "/home/lds/stable-diffusion.onnx/models/text_encoder/model.onnx"};
  std::vector<std::string> onnx_path = {text2image_param->model_value_[1]};
  infer_param->model_value_ = onnx_path;

  ClipGraph *clip_graph =
      new ClipGraph(name, {prompt, negative_prompt}, {output});
  dag::NodeDesc tokenize_desc("tokenize", {prompt->getName()}, {"token_ids"});
  dag::NodeDesc negative_tokenize_desc(
      "negative_tokenize", {negative_prompt->getName()}, {"token_ids"});
  dag::NodeDesc concat_desc("concat_desc",
                            {"prompt_ids", "negative_prompt_ids"},
                            {output->getName()});
  clip_graph->make(tokenize_desc, negative_tokenize_desc, concat_desc,
                   inference_type);
}

dag::Graph *createCLIPGraph(const std::string &name, dag::Edge *prompt,
                            dag::Edge *negative_prompt, dag::Edge *output,
                            base::InferenceType inference_type,
                            std::vector<base::Param *> &param) {
  dag::Graph *graph = new dag::Graph(name, {prompt, negative_prompt}, {output});

  Text2ImageParam *text2image_param = (Text2ImageParam *)param[0];

  dag::Edge *prompt_ids = graph->createEdge("prompt_ids");
  EmbeddingGraph *embedding_graph =
      (EmbeddingGraph *)(graph->createNode<EmbeddingGraph>(
          "embedding_subgraph", {prompt}, {prompt_ids}));
  embedding_graph->make(inference_type, "tokenizer");

  tokenizer::TokenizerPraram *tokenizer_param =
      new tokenizer::TokenizerPraram();
  tokenizer_param->tokenizer_type_ = tokenizer::TokenizerType::kTokenizerTypeHF;
  tokenizer_param->is_path_ = true;
  // tokenizer_param->json_blob_ =
  //     "/home/lds/stable-diffusion.onnx/models/tokenizer/tokenizer.json";
  tokenizer_param->json_blob_ = text2image_param->model_value_[0];
  embedding_graph->setTokenizerParam(tokenizer_param);

  inference::InferenceParam *infer_param = new inference::InferenceParam();
  infer_param->device_type_ = text2image_param->device_type_;
  infer_param->model_type_ = text2image_param->model_type_;
  infer_param->is_path_ = text2image_param->is_path_;
  // std::vector<std::string> onnx_path = {
  //     "/home/lds/stable-diffusion.onnx/models/text_encoder/model.onnx"};
  std::vector<std::string> onnx_path = {text2image_param->model_value_[1]};
  infer_param->model_value_ = onnx_path;
  embedding_graph->setInferParam(infer_param);

  dag::Edge *negative_prompt_ids = graph->createEdge("negative_prompt_ids");
  EmbeddingGraph *negative_embedding_graph =
      (EmbeddingGraph *)(graph->createNode<EmbeddingGraph>(
          "negative_embedding_subgraph", {negative_prompt},
          {negative_prompt_ids}));
  negative_embedding_graph->make(inference_type, "negative_tokenizer");
  negative_embedding_graph->setTokenizerParam(tokenizer_param);
  negative_embedding_graph->setInferParam(infer_param);

  ConCatNode *concat_node = (ConCatNode *)graph->createNode<ConCatNode>(
      "concat_node", {prompt_ids, negative_prompt_ids}, {output});
  DDIMSchedulerParam *scheduler_param = (DDIMSchedulerParam *)param[1];
  concat_node->setGuidance(scheduler_param->guidance_scale_);

  return graph;
}

}  // namespace stable_diffusion
}  // namespace nndeploy