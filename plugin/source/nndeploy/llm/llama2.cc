
#include "nndeploy/llm/llama2.h"

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
namespace llm {

/* parse config file */
LlmConfig parseConfig(const std::string& file_path) {
  LlmConfig config;

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

  // NNDEPLOY_LOGI("Graph params:\n");
  // NNDEPLOY_LOGI("layer_num=%d\n", config.layer_nums_);
  // NNDEPLOY_LOGI("hidden_size=%d\n", config.hidden_size_);
  // NNDEPLOY_LOGI("max_seq_len=%d\n", config.max_seq_len_);
  // NNDEPLOY_LOGI("model_value=%s\n", config.model_value_.c_str());
  // NNDEPLOY_LOGI("embedding_file=%s\n", config.embedding_file_.c_str());
  // NNDEPLOY_LOGI("tokenizer_json=%s\n", config.tokenizer_json_.c_str());
  // NNDEPLOY_LOGI("prompt_template=%s\n", config.prompt_template_.c_str());
  // NNDEPLOY_LOGI("kv.shape=[");
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

device::Tensor* EmbeddingNode::genPositionIds(int seq_len, int all_seq_len,
                                              base::DataType data_type,
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

device::Tensor* EmbeddingNode::genAttentionMask(int seq_len, int all_seq_len,
                                                base::DataType data_type,
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

device::Tensor* EmbeddingNode::genEmbedding(
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

base::Status EmbeddingNode::run() {
  base::Status status = base::kStatusCodeOk;

  EmbeddingParam* embedding_param = (EmbeddingParam*)(param_.get());
  bool is_prefill = embedding_param->is_prefill_;
  std::vector<std::vector<int32_t>> token_ids;
  if (is_prefill) {
    token_ids = ((tokenizer::TokenizerIds*)(inputs_[0]->getParam(this)))->ids_;
  } else {
    token_ids = embedding_param->token_ids_;
  }

  embedding_param->history_ids_.ids_ = token_ids;
  std::string embedding_file = embedding_param->embedding_file_;
  int hidden_size = embedding_param->hidden_size_;
  int seq_len = token_ids[0].size();
  int all_seq_len = embedding_param->all_seq_len_;
  past_kv_ = embedding_param->past_kv_;
  past_kv_->setName("past_key_values");

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

base::Status SampleNode::run() {
  base::Status status = base::kStatusCodeOk;

  device::Tensor* logits = inputs_[0]->getTensor(this);
  SampleParam* sample_params = dynamic_cast<SampleParam*>(param_.get());

  if (sample_params->is_prefill_) {
    tokenizer::TokenizerIds* token_ids =
        (tokenizer::TokenizerIds*)(sample_params->token_ids_->getParam(
            sample_params->emb_node_));
    sample_params->history_ids_.ids_ = token_ids->ids_;
  }
  std::vector<int32_t> history_ids = sample_params->history_ids_.ids_[0];

  int32_t out_token_id = sample(logits, history_ids);
  tokenizer::TokenizerIds* out_token = new tokenizer::TokenizerIds();

  /* when first time decode, append the prefill preditct token */
  if (is_first_ && !sample_params->is_prefill_) {
    out_token->ids_.push_back(
        {history_ids[history_ids.size() - 1], out_token_id});
    is_first_ = false;
  } else {
    out_token->ids_.push_back({out_token_id});
  }

  outputs_[0]->set(out_token, false);
  sample_params->history_ids_.ids_[0].push_back(out_token_id);

  return status;
}

int32_t SampleNode::sample(device::Tensor* logits,
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

base::Status LlmPrefillGraph::run() {
  base::Status status = base::kStatusCodeOk;

  setRunningFlag(true);
  EmbeddingParam* embedding_param =
      dynamic_cast<EmbeddingParam*>(prefill_embedding_node_->getParam());
  SampleParam* sample_params =
      dynamic_cast<SampleParam*>(prefill_sample_node_->getParam());

  status = executor_->run();
  NNDEPLOY_RETURN_ON_NEQ(status, base::kStatusCodeOk, "executor run failed!");
  history_ids_->ids_.push_back(sample_params->history_ids_.ids_[0]);

  setRunningFlag(false);
  return status;
}

void LlmPrefillGraph::genPastKeyValue() {
  device::Device* device = device::getDefaultHostDevice();
  device::TensorDesc past_kv_desc;
  past_kv_desc.data_type_ = base::dataTypeOf<float>();
  past_kv_desc.data_format_ = base::DataFormat::kDataFormatS1D;
  past_kv_desc.shape_ = kv_init_shape_;
  past_kv_ = new device::Tensor(device, past_kv_desc, "past_key_values");
}

void LlmPrefillGraph::createPrefillNodesEdges() {
  /* token node */
  prefill_token_ids_ = createEdge("prefill_token_ids");
  prefill_token_node_ = createNode<tokenizer::TokenizerEncodeCpp>(
      "token_node", {prompt_}, {prefill_token_ids_});

  /* embedding node */
  prefill_input_ids_ = createEdge("prefill_input_ids");
  prefill_attention_mask_ = createEdge("prefill_attention_mask");
  prefill_position_ids_ = createEdge("prefill_position_ids");
  prefill_past_key_values_ = createEdge("prefill_past_key_values");

  std::vector<dag::Edge*> embedding_in = {prefill_token_ids_};
  std::vector<dag::Edge*> embedding_out = {
      prefill_input_ids_, prefill_attention_mask_, prefill_position_ids_,
      prefill_past_key_values_};
  prefill_embedding_node_ =
      createNode<EmbeddingNode>("embedding_node", embedding_in, embedding_out);

  /* past kv */
  genPastKeyValue();

  /* prefill infer node */
  prefill_logits_ = createEdge("logits");
  std::vector<dag::Edge*> infer_out = {prefill_logits_, prefill_presents_};
  prefill_infer_node_ = createInfer<infer::Infer>(
      "prefill_infer", inference_type_, embedding_out, infer_out);

  /* sample node */
  prefill_sample_node_ = createNode<SampleNode>(
      "sample_node", {prefill_logits_}, {prefill_out_ids_});
}

void LlmPrefillGraph::setParams(bool is_path, base::ModelType model_type,
                                base::DeviceType device_type,
                                LlmConfig& config) {
  /* embedding params */
  EmbeddingParam* embedding_param =
      dynamic_cast<EmbeddingParam*>(prefill_embedding_node_->getParam());
  embedding_param->hidden_size_ = hidden_size_;
  embedding_param->embedding_file_ = config.embedding_file_;
  embedding_param->past_kv_ = past_kv_;
  embedding_param->is_prefill_ = true;

  /* token params */
  tokenizer::TokenizerPraram* token_param =
      dynamic_cast<tokenizer::TokenizerPraram*>(
          prefill_token_node_->getParam());
  token_param->is_path_ = false;
  token_param->tokenizer_type_ =
      nndeploy::tokenizer::TokenizerType::kTokenizerTypeHF;
  token_param->json_blob_ = config.tokenizer_json_;

  /* infer params */
  inference::InferenceParam* inference_param =
      (inference::InferenceParam*)(prefill_infer_node_->getParam());
  inference_param->is_path_ = true;
  inference_param->model_value_ = {config.model_value_};
  inference_param->device_type_ = device_type;
  inference_param->model_type_ = model_type;

  /* sample params */
  SampleParam* sample_params =
      dynamic_cast<SampleParam*>(prefill_sample_node_->getParam());
  sample_params->token_ids_ = prefill_token_ids_;
  sample_params->emb_node_ = prefill_embedding_node_;
  sample_params->is_prefill_ = true;
}

void LlmDecodeGraph::getStopTokens(std::string& token_file) {
  std::ifstream tok_file(token_file);
  if (!tok_file.good()) {
    printf("Failed: can't load tokenzier from: %s.\n", token_file.c_str());
    return;
  }

  std::string line;
  /*
    a little bit awkard, have to be clear about the content of tokenizer.txt
    stop number is in the second line
  */
  std::getline(tok_file, line);  // 1st line
  std::getline(tok_file, line);  // 2nd line
  std::istringstream line_str(line);
  int special_num, stop_num, prefix_num;
  line_str >> special_num >> stop_num >> prefix_num;
  std::getline(tok_file, line);
  std::istringstream specail_line(line);

  if (special_num) {
    // load special tokens
    special_tokens_.resize(special_num);
    for (int i = 0; i < special_num; i++) {
      specail_line >> special_tokens_[i];
    }
  }
  if (stop_num) {
    // load stop tokens
    stop_tokens_.resize(stop_num);
    for (int i = 0; i < stop_num; i++) {
      specail_line >> stop_tokens_[i];
    }
  }
}

void LlmDecodeGraph::createPrefillNodesEdges() {
  /* embedding node */
  decode_input_ids_ = createEdge("decode_input_ids");
  decode_attention_mask_ = createEdge("decode_attention_mask");
  decode_position_ids_ = createEdge("decode_position_ids");
  decode_past_key_values_ = createEdge("decode_past_key_values");

  std::vector<dag::Edge*> embedding_in = {decode_embedding_ids_,
                                          decode_prefill_kv_};
  std::vector<dag::Edge*> embedding_out = {
      decode_input_ids_, decode_attention_mask_, decode_position_ids_,
      decode_past_key_values_};
  decode_embedding_node_ =
      createNode<EmbeddingNode>("embedding_node", embedding_in, embedding_out);

  /* decode infer node */
  /* use "new" to create duplicated name edge */
  decode_logits_ = new dag::Edge("logits");
  decode_presents_ = new dag::Edge("presents");

  std::vector<dag::Edge*> infer_out = {decode_logits_, decode_presents_};
  decode_infer_node_ = createInfer<infer::Infer>(
      "decode_infer", inference_type_, embedding_out, infer_out);

  /* sample node */
  decode_out_ids_ = createEdge("decode_out_ids");
  decode_sample_node_ =
      createNode<SampleNode>("sample_node", decode_logits_, decode_out_ids_);

  /* decode node */
  decode_node_ = createNode<tokenizer::TokenizerDecodeCpp>(
      "decode_node", {decode_out_ids_}, {decode_out_words_});
}

void LlmDecodeGraph::setParams(bool is_path, base::ModelType model_type,
                               base::DeviceType device_type,
                               LlmConfig& config) {
  /* embedding params */
  EmbeddingParam* embedding_param =
      dynamic_cast<EmbeddingParam*>(decode_embedding_node_->getParam());
  embedding_param->embedding_file_ = config.embedding_file_;
  embedding_param->hidden_size_ = hidden_size_;
  embedding_param->gen_seq_len_ = 1;
  embedding_param->past_kv_ = past_kv_;
  embedding_param->is_prefill_ = false;

  /* infer params */
  inference::InferenceParam* inference_param =
      (inference::InferenceParam*)(decode_infer_node_->getParam());
  inference_param->is_path_ = true;
  inference_param->model_value_ = {config.model_value_};
  inference_param->device_type_ = device_type;
  inference_param->model_type_ = model_type;

  /* sample params */
  SampleParam* sample_params =
      dynamic_cast<SampleParam*>(decode_sample_node_->getParam());
  sample_params->token_ids_ = decode_embedding_ids_;
  sample_params->emb_node_ = decode_embedding_node_;
  sample_params->stop_tokens_.ids_.push_back(stop_tokens_);
  sample_params->is_prefill_ = false;

  /* decode token params */
  tokenizer::TokenizerPraram* token_param =
      dynamic_cast<tokenizer::TokenizerPraram*>(decode_node_->getParam());
  token_param->is_path_ = false;
  token_param->tokenizer_type_ =
      nndeploy::tokenizer::TokenizerType::kTokenizerTypeHF;
  token_param->json_blob_ = config.tokenizer_json_;
}

int LlmDecodeGraph::loops() {
  /* simply return maximum seqence length */
  return max_seq_len_;
}

base::Status LlmDecodeGraph::run() {
  base::Status status = base::kStatusCodeOk;

  setRunningFlag(true);

  int iters = loops();
  for (int i = 0; i < iters; ++i) {
    if (isStop()) {
      tokenizer::TokenizerText* word =
          (tokenizer::TokenizerText*)(decode_out_words_->getParam(
              decode_node_));
      word->texts_[0] = result_;
      break;
    }

    /* embedding params */
    EmbeddingParam* embedding_param =
        dynamic_cast<EmbeddingParam*>(decode_embedding_node_->getParam());
    embedding_param->history_ids_ = *history_ids_;
    SampleParam* sample_param =
        dynamic_cast<SampleParam*>(decode_sample_node_->getParam());

    if (!is_first_) {
      embedding_param->all_seq_len_++;
    } else {
      /* first time decode will fetch ids and past_key_values from prefill graph
       */
      tokenizer::TokenizerIds* prefill_out_token_id =
          (tokenizer::TokenizerIds*)(decode_embedding_ids_->getParam(
              decode_embedding_node_));
      embedding_param->past_kv_ =
          decode_prefill_kv_->getTensor(decode_embedding_node_);
      embedding_param->token_ids_ = prefill_out_token_id->ids_;
      sample_param->history_ids_ = *history_ids_;
      embedding_param->all_seq_len_ =
          embedding_param->history_ids_.ids_[0].size();
      is_first_ = false;
    }
    embedding_param->gen_seq_len_++;

    status = executor_->run();
    NNDEPLOY_RETURN_ON_NEQ(status, base::kStatusCodeOk, "executor runfailed!");

    /* close loop input and output */
    tokenizer::TokenizerIds* token_id =
        (tokenizer::TokenizerIds*)(decode_out_ids_->getParam(
            decode_sample_node_));
    *(embedding_param->past_kv_) =
        *(decode_presents_->getTensor(decode_infer_node_));

    /* first time decode ==> size of token_id->ids_[0] is 2 */
    if (token_id->ids_[0].size() == 2) {
      embedding_param->token_ids_[0] = {token_id->ids_[0][1]};
    } else {
      embedding_param->token_ids_ = token_id->ids_;
    }

    history_ids_->ids_ = sample_param->history_ids_.ids_;
    tokenizer::TokenizerText* word =
        (tokenizer::TokenizerText*)(decode_out_words_->getParam(decode_node_));

    /* check whether token_id is <|im_end|> */
    int history_size = history_ids_->ids_[0].size();
    int out_token_id = history_ids_->ids_[0][history_size - 1];
    bool not_append =
        std::find(sample_param->stop_tokens_.ids_[0].begin(),
                  sample_param->stop_tokens_.ids_[0].end(),
                  out_token_id) != sample_param->stop_tokens_.ids_[0].end();
    if (!not_append) result_.append(word->texts_[0]);
  }

  setRunningFlag(false);
  return status;
}

dag::TypeGraphRegister g_register_llama2_graph(NNDEPLOY_LLAMA2,
                                               createLlmLlama2Graph);

dag::Graph* createLlmLlama2Graph(const std::string& name,
                                 base::InferenceType inference_type,
                                 base::DeviceType device_type,
                                 dag::Edge* prompt, dag::Edge* out,
                                 base::ModelType model_type, bool is_path,
                                 std::vector<std::string> config_path) {
  LlmConfig config = parseConfig(config_path[0]);
  dag::Graph* llama2_graph = new dag::Graph(name, {prompt}, {out});

  /* create prefill graph */
  dag::Edge* prefill_out_ids = new dag::Edge("prefill_out_ids");
  dag::Edge* prefill_presents = new dag::Edge("presents");
  std::vector<dag::Edge*> prefill_in = {prompt};
  std::vector<dag::Edge*> prefill_out = {prefill_out_ids, prefill_presents};
  LlmPrefillGraph* prefill_graph = new LlmPrefillGraph(
      "llama2_prefill", prefill_in, prefill_out, inference_type, device_type,
      model_type, is_path, config);
  llama2_graph->addNode(prefill_graph);
  std::vector<dag::Edge*> decode_out = {out};
  LlmDecodeGraph* decode_graph = new LlmDecodeGraph(
      "llama2_decode", prefill_out, decode_out, prefill_graph->past_kv_,
      prefill_graph->history_ids_, inference_type, device_type, model_type,
      is_path, config);
  llama2_graph->addNode(decode_graph);

  return llama2_graph;
}

REGISTER_NODE("nndeploy::llm::EmbeddingNode", EmbeddingNode);
REGISTER_NODE("nndeploy::llm::SampleNode", SampleNode);
REGISTER_NODE("nndeploy::llm::PromptNode", PromptNode);
REGISTER_NODE("nndeploy::llm::LlmPrefillGraph", LlmPrefillGraph);
REGISTER_NODE("nndeploy::llm::LlmDecodeGraph", LlmDecodeGraph);

}  // namespace llm
}  // namespace nndeploy
