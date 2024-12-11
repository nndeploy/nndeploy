
#include "nndeploy/llm/llama2.h"

#include "nndeploy/base/any.h"
#include "nndeploy/base/common.h"
#include "nndeploy/base/glic_stl_include.h"
#include "nndeploy/base/log.h"
#include "nndeploy/base/macro.h"
#include "nndeploy/base/object.h"
#include "nndeploy/base/status.h"
#include "nndeploy/base/string.h"
#include "nndeploy/llm/util.h"
#include "nndeploy/dag/edge.h"
#include "nndeploy/dag/node.h"
#include "nndeploy/device/buffer.h"
#include "nndeploy/device/device.h"
#include "nndeploy/device/memory_pool.h"
#include "nndeploy/device/tensor.h"
#include "nndeploy/infer/infer.h"
#include "nndeploy/op/op_softmax.h"
#include "nndeploy/tokenizer/tokenizer_cpp/tokenizer_cpp.h"
#include "nndeploy/tokenizer/tokenizer.h"

namespace nndeploy {
namespace llm {

dag::TypeGraphRegister g_register_llama2_graph(NNDEPLOY_LLAMA2,
                                               createLlmLlama2Graph);

base::Status Llama2PostProcess::run() {
  Llama2PostParam *param = (Llama2PostParam *)param_.get();

  device::Tensor *tensor = inputs_[0]->getTensor(this);
  std::shared_ptr<ir::SoftmaxParam> op_param =
      std::make_shared<ir::SoftmaxParam>();
  op_param->axis_ = 1;
  device::Tensor softmax_tensor(tensor->getDevice(), tensor->getDesc());
  base::Status status = op::softmax(tensor, op_param, &softmax_tensor);
  if (status != base::kStatusCodeOk) {
    NNDEPLOY_LOGE("op::softmax failed!\n");
    return status;
  }
  float *data = (float *)softmax_tensor.getData();
  int batch = softmax_tensor.getShapeIndex(0);
  int num_classes = softmax_tensor.getShapeIndex(1);

  LlmResult *results = new LlmResult();
  param->topk_ = std::min(num_classes, param->topk_);
  int topk = param->topk_;
  results->labels_.resize(topk * batch);

  // 使用优先队列找出topk个最大值
  for (int b = 0; b < batch; ++b) {
    float *iter_data = data + b * num_classes;
    std::vector<int> label_ids_ = topKIndices(iter_data, num_classes, topk);
    ;
    for (int i = 0; i < topk; ++i) {
      results->labels_[i + b * topk].index_ = b;
      results->labels_[i + b * topk].label_ids_ = label_ids_[i];
      results->labels_[i + b * topk].scores_ = *(iter_data + label_ids_[i]);
    }
  }

  outputs_[0]->set(results, inputs_[0]->getIndex(this), false);
  return base::kStatusCodeOk;
}

dag::Graph *createLlmLlama2Graph(
    const std::string &name, base::InferenceType inference_type,
    base::DeviceType device_type, dag::Edge *input, dag::Edge *output,
    base::ModelType model_type, bool is_path,
    std::vector<std::string> model_value) {

  dag::Graph *graph = new dag::Graph(name, input, output);
  //dag::Edge *infer_input = graph->createEdge("data");
  //dag::Edge *infer_output = graph->createEdge("resnetv17_dense0_fwd");
  dag::Edge *infer_input = graph->createEdge("input_ids");
  dag::Edge *infer_output = graph->createEdge("logits");

  // prompts --> preprocess --> token_ids
  //dag::Node *pre = graph->createNode<preprocess::CvtColorResize>(
  dag::Node *pre = graph->createNode<tokenizer::TokenizerCpp>(
      "preprocess", input, infer_input);

  // token_ids --> graph --> logits
  dag::Node *infer = graph->createInfer<infer::Infer>(
      "infer", inference_type, infer_input, infer_output);

  // logits --> postprocess --> pred_tokens
  dag::Node *post = graph->createNode<Llama2PostProcess>(
      "postprocess", infer_output, output);

  tokenizer::TokenizerPraram *pre_param =
      dynamic_cast<tokenizer::TokenizerPraram*>(pre->getParam());
  pre_param->is_encode_ = true;
  pre_param->is_path_ = false;
  pre_param->tokenizer_type_ = nndeploy::tokenizer::TokenizerType::kTokenizerTypeHF;
  pre_param->json_blob_ = "/home/raymond/workspace/llm-export/Qwen2-0.5B-Instruct/tokenizer.json";
  pre_param->model_blob_ = "";
  pre_param->vocab_blob_ = "";
  pre_param->merges_blob_ = "";
  pre_param->added_tokens_ = "";
  pre_param->max_length_ = 256;

  inference::InferenceParam *inference_param =
      (inference::InferenceParam *)(infer->getParam());
  inference_param->is_path_ = is_path;
  inference_param->model_value_ = model_value;
  inference_param->device_type_ = device_type;
  inference_param->model_type_ = model_type;

  // TODO: 很多信息可以从 preprocess 和 infer 中获取
  Llama2PostParam *post_param =
      dynamic_cast<Llama2PostParam *>(post->getParam());
  post_param->topk_ = 1;

  return graph;
}

}  // namespace llm 
}  // namespace nndeploy
