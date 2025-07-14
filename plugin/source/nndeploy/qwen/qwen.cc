
#include "nndeploy/qwen/qwen.h"

#include <fstream>
#include <iostream>

#include "nndeploy/base/any.h"
#include "nndeploy/base/common.h"
#include "nndeploy/base/glic_stl_include.h"
#include "nndeploy/base/log.h"
#include "nndeploy/base/macro.h"
#include "nndeploy/base/object.h"
#include "nndeploy/base/status.h"
#include "nndeploy/base/string.h"
#include "nndeploy/dag/edge.h"
#include "nndeploy/dag/node.h"
#include "nndeploy/device/buffer.h"
#include "nndeploy/device/device.h"
#include "nndeploy/device/memory_pool.h"
#include "nndeploy/device/tensor.h"
#include "nndeploy/infer/infer.h"
#include "nndeploy/op/op_softmax.h"
#include "nndeploy/tokenizer/tokenizer.h"
#include "nndeploy/tokenizer/tokenizer_cpp/tokenizer_cpp.h"
#include "rapidjson/document.h"
#include "rapidjson/filereadstream.h"

namespace nndeploy {
namespace qwen {

/* parse config file */
QwenConfig parseConfig(const std::string& file_path) {
  QwenConfig config;

  std::ifstream ifs(file_path);
  if (!ifs.is_open()) {
    NNDEPLOY_LOGE("Could not open %s:\n", file_path.c_str());
    return config;
  }

  // read json data to content
  std::string content((std::istreambuf_iterator<char>(ifs)),
                      std::istreambuf_iterator<char>());
  ifs.close();

  rapidjson::Document llm_config;
  llm_config.Parse(content.c_str());
  if (llm_config.HasParseError()) {
    NNDEPLOY_LOGE("Error parsing JSON file%s:\n", file_path.c_str());
    return config;
  }

  // parse data
  config.layer_nums_ = llm_config["layer_nums"].GetInt();
  config.hidden_size_ = llm_config["hidden_size"].GetInt();
  config.max_seq_len_ = llm_config["max_seq_len"].GetInt();

  config.model_value_ = llm_config["model_path"].GetString();
  config.embedding_file_ = llm_config["embedding_file"].GetString();
  config.tokenizer_json_ = llm_config["tokenizer_json"].GetString();
  config.tokenizer_txt_ = llm_config["tokenizer_txt"].GetString();
  config.prompt_template_ = llm_config["prompt_template"].GetString();
  config.prompt_ = llm_config["prompt"].GetString();

  rapidjson::Value& key_value_shape = llm_config["key_value_shape"];
  for (size_t i = 0; i < key_value_shape.Size(); i++) {
    config.kv_init_shape_.push_back(key_value_shape[i].GetInt());
  }

  config.kv_init_shape_.insert(config.kv_init_shape_.begin(),
                               config.layer_nums_);
  for (auto& s : config.kv_init_shape_) NNDEPLOY_LOGI("%d,", s);
  NNDEPLOY_LOGI("]\n");

  return config;
}

std::string PromptNode::applyTemplate(std::string prompt_template,
                                      const std::string& content,
                                      const std::string& role) {
  if (prompt_template.empty()) return content;
  if (!role.empty()) {
    const std::string placeholder = "%r";
    size_t start_pos = prompt_template.find(placeholder);
    if (start_pos == std::string::npos) return content;
    prompt_template.replace(start_pos, placeholder.length(), role);
  }
  const std::string placeholder = "%s";
  size_t start_pos = prompt_template.find(placeholder);
  if (start_pos == std::string::npos) return content;
  prompt_template.replace(start_pos, placeholder.length(), content);
  return prompt_template;
}

base::Status PromptNode::run() {
  PromptParam* prompt_params = (PromptParam*)this->getParam();
  std::string template_prompt = applyTemplate(prompt_params->prompt_template_,
                                              prompt_params->user_content_);
  tokenizer::TokenizerText* prompt = new tokenizer::TokenizerText();
  prompt->texts_.emplace_back(template_prompt);
  outputs_[0]->set(prompt, false);
  return base::kStatusCodeOk;
}

device::Tensor* PrefillEmbeddingNode::genPositionIds(
    int seq_len, int all_seq_len, base::DataType data_type,
    base::DataFormat data_format) {
  /* create position_ids tensor */
  device::Device* device = device::getDefaultHostDevice();
  device::TensorDesc position_ids_desc;
  position_ids_desc.data_type_ = data_type;
  position_ids_desc.data_format_ = data_format;
  position_ids_desc.shape_ = {1, seq_len};
  device::Tensor* position_ids;
  if (is_first_) {
    position_ids =
        new device::Tensor(device, position_ids_desc, "position_ids");
  } else {
    position_ids = outputs_[2]->getTensor(this);
  }

  auto ptr = (int*)position_ids->getData();
  if (seq_len == 1) {
    ptr[0] = all_seq_len;
  } else {
    for (int i = 0; i < seq_len; i++) {
      ptr[i] = i + all_seq_len;
    }
  }

  return position_ids;
}

device::Tensor* PrefillEmbeddingNode::genAttentionMask(
    int seq_len, int all_seq_len, base::DataType data_type,
    base::DataFormat data_format) {
  int kv_seq_len = all_seq_len + seq_len;
  if (seq_len == 1) kv_seq_len = seq_len;

  /* create attetion_mask tensor */
  device::Device* device = device::getDefaultHostDevice();
  device::TensorDesc attention_mask_desc;
  attention_mask_desc.data_type_ = data_type;
  attention_mask_desc.data_format_ = data_format;
  attention_mask_desc.shape_ = {1, 1, seq_len, kv_seq_len};
  device::Tensor* attention_mask;
  if (is_first_) {
    attention_mask =
        new device::Tensor(device, attention_mask_desc, "attention_mask");
  } else {
    attention_mask = outputs_[1]->getTensor(this);
  }

  auto ptr = (float*)attention_mask->getData();
  for (int i = 0; i < seq_len; i++) {
    for (int j = 0; j < kv_seq_len; j++) {
      int row = i + all_seq_len;
      ptr[kv_seq_len * i + j] =
          (j > row) * std::numeric_limits<float>::lowest();
    }
  }

  return attention_mask;
}

device::Tensor* PrefillEmbeddingNode::genEmbedding(
    const std::vector<int32_t>& input_ids, int seq_len, int hidden_size,
    base::DataType data_type, base::DataFormat data_format,
    std::string& embedding_file) {
  /* create input_id tensor */
  device::Device* device = device::getDefaultHostDevice();
  device::TensorDesc input_id_desc;
  input_id_desc.data_type_ = data_type;
  input_id_desc.data_format_ = data_format;
  input_id_desc.shape_ = {seq_len, 1, hidden_size};
  device::Tensor* inputs_embeds;
  if (is_first_) {
    inputs_embeds = new device::Tensor(device, input_id_desc, "input_ids");
  } else {
    inputs_embeds = outputs_[0]->getTensor(this);
  }

  size_t size = hidden_size * sizeof(int16_t);
  FILE* file = fopen(embedding_file.c_str(), "rb");
  std::unique_ptr<int16_t[]> buffer(new int16_t[hidden_size]);
  for (size_t i = 0; i < seq_len; i++) {
    fseek(file, input_ids[i] * size, SEEK_SET);
    size_t bytes_read = fread(buffer.get(), 1, size, file);
    (void)bytes_read;
    auto ptr = (int16_t*)inputs_embeds->getData() + i * hidden_size * 2;
    for (int j = 0; j < hidden_size; j++) {
      ptr[j * 2] = 0;
      ptr[j * 2 + 1] = buffer[j];
    }
  }
  fclose(file);
  return inputs_embeds;
}

base::Status PrefillEmbeddingNode::run() {
  base::Status status = base::kStatusCodeOk;

  EmbeddingParam* embedding_param = (EmbeddingParam*)(param_.get());
  bool is_prefill = embedding_param->is_prefill_;
  std::vector<std::vector<int32_t>> token_ids;
  token_ids = ((tokenizer::TokenizerIds*)(inputs_[0]->getParam(this)))->ids_;

  std::string embedding_file = embedding_param->embedding_file_;
  int hidden_size = embedding_param->hidden_size_;
  int seq_len = token_ids[0].size();
  int all_seq_len = embedding_param->all_seq_len_;

  /* considering sample node will update past_kv,
    past_kv has already been set in create_prefill_nodes_edges */
  auto inputs_embeds = genEmbedding(
      token_ids[0], seq_len, hidden_size, embedding_param->data_type_,
      embedding_param->data_format_, embedding_file);

  auto attention_mask =
      genAttentionMask(seq_len, all_seq_len, embedding_param->data_type_,
                       embedding_param->data_format_);

  auto position_ids =
      genPositionIds(seq_len, all_seq_len, embedding_param->posid_data_type_,
                     embedding_param->data_format_);

  if (is_first_) is_first_ = false;
  outputs_[0]->set(inputs_embeds, false);
  outputs_[1]->set(attention_mask, false);
  outputs_[2]->set(position_ids, false);
  outputs_[3]->set(past_kv_, false);

  return status;
}

base::Status PrefillSampleNode::run() {
  base::Status status = base::kStatusCodeOk;

  device::Tensor* logits = inputs_[0]->getTensor(this);
  tokenizer::TokenizerIds* token_ids =
      (tokenizer::TokenizerIds*)inputs_[1]->getParam(this);
  SampleParam* sample_params = dynamic_cast<SampleParam*>(param_.get());
  std::vector<int32_t> history_ids = token_ids->ids_[0];

  int32_t out_token_id = sample(logits, history_ids);
  tokenizer::TokenizerIds* out_token = new tokenizer::TokenizerIds();

  out_token->ids_.push_back({out_token_id});

  outputs_[0]->set(out_token, false);

  return status;
}

base::Status DecodeSampleNode::run() {
  base::Status status = base::kStatusCodeOk;

  device::Tensor* logits = inputs_[0]->getTensor(this);
  SampleParam* sample_params = dynamic_cast<SampleParam*>(param_.get());

  if (is_first_) {
    tokenizer::TokenizerIds* token_ids =
        (tokenizer::TokenizerIds*)inputs_[1]->getParam(this);
    history_ids_ = token_ids->ids_[0];
  }

  int32_t out_token_id = sample(logits, history_ids_);
  tokenizer::TokenizerIds* out_token = new tokenizer::TokenizerIds();

  /* when first time decode, append the prefill preditct token */
  if (is_first_ && !sample_params->is_prefill_) {
    out_token->ids_.push_back(
        {history_ids_[history_ids_.size() - 1], out_token_id});
    is_first_ = false;
  } else {
    out_token->ids_.push_back({out_token_id});
  }

  outputs_[0]->set(out_token, false);
  history_ids_.push_back(out_token_id);

  return status;
}

int32_t PrefillSampleNode::sample(device::Tensor* logits,
                                  const std::vector<int>& history_ids) {
  std::unordered_set<int> ids_set(history_ids.begin(), history_ids.end());
  auto scores = (float*)logits->getData();
  auto shape = logits->getShape();
  auto size = std::accumulate(shape.begin(), shape.end(), 1,
                              std::multiplies<int64_t>());
  // repetition penalty
  const float repetition_penalty = 1.1;
  for (auto id : ids_set) {
    float score = scores[id];
    scores[id] =
        score < 0 ? score * repetition_penalty : score / repetition_penalty;
  }
  // argmax
  float max_score = 0;
  int token_id = 0;
  for (int i = 0; i < size; i++) {
    float score = scores[i];
    if (score > max_score) {
      max_score = score;
      token_id = i;
    }
  }

  return token_id;
}

int32_t DecodeSampleNode::sample(device::Tensor* logits,
                                 const std::vector<int>& history_ids) {
  std::unordered_set<int> ids_set(history_ids.begin(), history_ids.end());
  auto scores = (float*)logits->getData();
  auto shape = logits->getShape();
  auto size = std::accumulate(shape.begin(), shape.end(), 1,
                              std::multiplies<int64_t>());
  // repetition penalty
  const float repetition_penalty = 1.1;
  for (auto id : ids_set) {
    float score = scores[id];
    scores[id] =
        score < 0 ? score * repetition_penalty : score / repetition_penalty;
  }
  // argmax
  float max_score = 0;
  int token_id = 0;
  for (int i = 0; i < size; i++) {
    float score = scores[i];
    if (score > max_score) {
      max_score = score;
      token_id = i;
    }
  }

  return token_id;
}

dag::Graph* createQwenGraph(const std::string& name,
                            base::InferenceType inference_type,
                            base::DeviceType device_type, dag::Edge* prompt,
                            dag::Edge* out, base::ModelType model_type,
                            bool is_path,
                            std::vector<std::string> config_path) {
  QwenConfig config = parseConfig(config_path[0]);
  dag::Graph* llm_graph = new dag::Graph(name, {prompt}, {out});

  dag::Edge* prefill_out_ids = llm_graph->createEdge("prefill_out_ids");
  dag::Edge* prefill_presents = llm_graph->createEdge("presents");
  dag::Edge* history_ids = llm_graph->createEdge("history_ids");

  std::vector<dag::Edge*> prefill_in = {prompt};
  std::vector<dag::Edge*> prefill_out = {prefill_out_ids, prefill_presents,
                                         history_ids};
  QwenPrefill* prefill = new QwenPrefill("prefill", prefill_in, prefill_out);
  llm_graph->addNode(prefill);

  std::vector<dag::Edge*> decode_out = {out};
  QwenDecode* decode = new QwenDecode("decode", prefill_out, decode_out);
  llm_graph->addNode(decode);

  return llm_graph;
}

// REGISTER_NODE("nndeploy::llm::EmbeddingNode", EmbeddingNode);
// REGISTER_NODE("nndeploy::llm::SampleNode", SampleNode);
// REGISTER_NODE("nndeploy::llm::PromptNode", PromptNode);
// REGISTER_NODE("nndeploy::llm::LlmPrefillGraph", LlmPrefillGraph);
// REGISTER_NODE("nndeploy::llm::LlmDecodeGraph", LlmDecodeGraph);

}  // namespace qwen
}  // namespace nndeploy
