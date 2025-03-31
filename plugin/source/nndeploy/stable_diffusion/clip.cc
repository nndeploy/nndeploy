#include "nndeploy/stable_diffusion/clip.h"

#include "nndeploy/infer/infer.h"
#include "nndeploy/op/op_concat.h"
#include "nndeploy/stable_diffusion/ddim_scheduler.h"
#include "nndeploy/stable_diffusion/utils.h"
#include "nndeploy/tokenizer/tokenizer_cpp/tokenizer_cpp.h"

namespace nndeploy {
namespace stable_diffusion {

class NNDEPLOY_CC_API CvtTokenIds2TensorNode : public dag::Node {
 public:
  CvtTokenIds2TensorNode(const std::string &name,
                         std::vector<dag::Edge *> inputs,
                         std::vector<dag::Edge *> outputs)
      : dag::Node(name, inputs, outputs) {}

  virtual ~CvtTokenIds2TensorNode() {}

  virtual base::Status run() {
    int index = this->getInput(0)->getIndex(this);
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
    device::Tensor *output = this->getOutput(0)->create(device, desc, index);
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

class NNDEPLOY_CC_API ConCatNode : public dag::Node {
 public:
  ConCatNode(const std::string &name, std::vector<dag::Edge *> inputs,
             std::vector<dag::Edge *> outputs)
      : dag::Node(name, inputs, outputs) {}

  virtual ~ConCatNode(){};

  base::Status setGuidance(float guidance) {
    guidance_ = guidance;
    return base::kStatusCodeOk;
  }

  virtual base::Status run() {
    bool do_classifier_free_guidance = (guidance_ > 1.0) ? true : false;

    int index = this->getInput(0)->getIndex(this);
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
      output = this->getOutput(0)->create(device, desc, index);

      std::shared_ptr<ir::ConcatParam> param =
          std::make_shared<ir::ConcatParam>();
      param->axis_ = 0;
      op::concat({negative_prompt, prompt}, param, output);
    } else {
      device::TensorDesc desc(base::dataTypeOf<float>(), base::kDataFormatNCL,
                              shape);
      output = this->getOutput(0)->create(device, desc, index);
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
      : dag::Graph(name, inputs, outputs) {}
  ~EmbeddingGraph(){};

  base::Status setTokenizerParam(tokenizer::TokenizerPraram *param) {
    tokenizer_node_->setParam(param);
    return base::kStatusCodeOk;
  }

  base::Status setInferParam(inference::InferenceParam *param) {
    clip_infer_node_->setParam(param);
    return base::kStatusCodeOk;
  }

  base::Status make(base::InferenceType inference_type, std::string name) {
    prompt_ = this->getInput(0);
    if (prompt_ == nullptr) {
      NNDEPLOY_LOGE("prompt is nullptr\n");
      return base::kStatusCodeErrorInvalidParam;
    }
    token_ids_ = this->createEdge("token_ids_");
    tokenizer_node_ = (tokenizer::TokenizerEncodeCpp *)this
                          ->createNode<tokenizer::TokenizerEncodeCpp>(
                              name, {prompt_}, {token_ids_});

    infer_ids_ = this->createEdge("infer_ids");
    cvt_node_ =
        (CvtTokenIds2TensorNode *)this->createNode<CvtTokenIds2TensorNode>(
            "cvt_token_ids_2_tensor", {token_ids_}, {infer_ids_});

    embedding_ = this->getOutput(0);
    if (embedding_ == nullptr) {
      NNDEPLOY_LOGE("embedding is nullptr\n");
      return base::kStatusCodeErrorInvalidParam;
    }
    clip_infer_node_ = (infer::Infer *)this->createInfer<infer::Infer>(
        "clip_infer", inference_type, {infer_ids_}, {embedding_});
    return base::kStatusCodeOk;
  }

 private:
  dag::Edge *prompt_;
  dag::Node *tokenizer_node_;
  dag::Edge *token_ids_;
  dag::Node *cvt_node_;
  dag::Edge *infer_ids_;
  dag::Node *clip_infer_node_;
  dag::Edge *embedding_;
};

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