
#include "nndeploy/qwen/qwen.h"

#include <limits.h>
#include <sys/stat.h>
// #include <unistd.h>

#include <ctime>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>

#include "nndeploy/base/any.h"
#include "nndeploy/base/common.h"
#include "nndeploy/base/file.h"
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

// void printTokenizerIds(const std::vector<std::vector<int32_t>> token_ids,
//                        std::string title, std::ostream& os = std::cout) {
//   for (size_t seq_idx = 0; seq_idx < token_ids.size(); ++seq_idx) {
//     os << title << " " << "Seq " << seq_idx << ": ";
//     for (int id : token_ids[seq_idx]) {
//       os << id << ' ';
//     }
//     os << '\n';
//   }
// }

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

/**
 * @brief PromptParam Serialize Function
 */
base::Status PromptParam::serialize(
    rapidjson::Value& json, rapidjson::Document::AllocatorType& allocator) {
  this->addRequiredParam("user_content_");
  base::Status status = base::Param::serialize(json, allocator);
  rapidjson::Value prompt_template_val;
  prompt_template_val.SetString(
      prompt_template_.c_str(),
      static_cast<rapidjson::SizeType>(prompt_template_.length()), allocator);
  json.AddMember("prompt_template_", prompt_template_val, allocator);

  rapidjson::Value user_content_val;
  user_content_val.SetString(
      user_content_.c_str(),
      static_cast<rapidjson::SizeType>(user_content_.length()), allocator);
  json.AddMember("user_content_", user_content_val, allocator);

  return base::kStatusCodeOk;
}

/**
 * @brief PromptParam Deserialize Function
 */
base::Status PromptParam::deserialize(rapidjson::Value& json) {
  base::Status status = base::Param::deserialize(json);
  if (status != base::kStatusCodeOk) {
    NNDEPLOY_LOGE("ResizeParam::deserialize failed\n");
    return status;
  }
  if (json.HasMember("prompt_template_") &&
      json["prompt_template_"].IsString()) {
    prompt_template_ = json["prompt_template_"].GetString();
  }
  if (json.HasMember("user_content_") && json["user_content_"].IsString()) {
    user_content_ = json["user_content_"].GetString();
  }

  return base::kStatusCodeOk;
}

/**
 * @brief PrefillEmbeddingParam Serialize Function
 */
base::Status PrefillEmbeddingParam::serialize(
    rapidjson::Value& json, rapidjson::Document::AllocatorType& allocator) {
  json.AddMember("hidden_size_", hidden_size_, allocator);
  json.AddMember("all_seq_len_", all_seq_len_, allocator);
  json.AddMember("gen_seq_len_", gen_seq_len_, allocator);

  rapidjson::Value embedding_file_val;
  embedding_file_val.SetString(
      embedding_file_.c_str(),
      static_cast<rapidjson::SizeType>(embedding_file_.length()), allocator);
  json.AddMember("embedding_file_", embedding_file_val, allocator);

  return base::kStatusCodeOk;
}

/**
 * @brief PrefillEmbeddingParam Deserialize Function
 */
base::Status PrefillEmbeddingParam::deserialize(rapidjson::Value& json) {
  if (json.HasMember("hidden_size_") && json["hidden_size_"].IsInt()) {
    hidden_size_ = json["hidden_size_"].GetInt();
  }
  if (json.HasMember("all_seq_len_") && json["all_seq_len_"].IsInt()) {
    all_seq_len_ = json["all_seq_len_"].GetInt();
  }
  if (json.HasMember("gen_seq_len_") && json["gen_seq_len_"].IsInt()) {
    gen_seq_len_ = json["gen_seq_len_"].GetInt();
  }
  if (json.HasMember("embedding_file_") && json["embedding_file_"].IsString()) {
    embedding_file_ = json["embedding_file_"].GetString();
  }
  return base::kStatusCodeOk;
}

/**
 * @brief DecodeEmbeddingParam Serialize Function
 */
base::Status DecodeEmbeddingParam::serialize(
    rapidjson::Value& json, rapidjson::Document::AllocatorType& allocator) {
  json.AddMember("hidden_size_", hidden_size_, allocator);
  json.AddMember("all_seq_len_", all_seq_len_, allocator);
  json.AddMember("gen_seq_len_", gen_seq_len_, allocator);

  rapidjson::Value embedding_file_val;
  embedding_file_val.SetString(
      embedding_file_.c_str(),
      static_cast<rapidjson::SizeType>(embedding_file_.length()), allocator);
  json.AddMember("embedding_file_", embedding_file_val, allocator);

  return base::kStatusCodeOk;
}

/**
 * @brief DecodeEmbeddingParam Deserialize Function
 */
base::Status DecodeEmbeddingParam::deserialize(rapidjson::Value& json) {
  if (json.HasMember("hidden_size_") && json["hidden_size_"].IsInt()) {
    hidden_size_ = json["hidden_size_"].GetInt();
  }
  if (json.HasMember("all_seq_len_") && json["all_seq_len_"].IsInt()) {
    all_seq_len_ = json["all_seq_len_"].GetInt();
  }
  if (json.HasMember("gen_seq_len_") && json["gen_seq_len_"].IsInt()) {
    gen_seq_len_ = json["gen_seq_len_"].GetInt();
  }
  if (json.HasMember("embedding_file_") && json["embedding_file_"].IsString()) {
    embedding_file_ = json["embedding_file_"].GetString();
  }
  return base::kStatusCodeOk;
}

base::Status PrintNode::serialize(
    rapidjson::Value& json, rapidjson::Document::AllocatorType& allocator) {
  this->addRequiredParam("path_");
  base::Status status = dag::Node::serialize(json, allocator);
  if (status != base::kStatusCodeOk) {
    return status;
  }
  rapidjson::Value path_val;
  path_val.SetString(path_.c_str(),
                     static_cast<rapidjson::SizeType>(path_.length()),
                     allocator);
  json.AddMember("path_", path_val, allocator);
  return base::kStatusCodeOk;
}

base::Status PrintNode::deserialize(rapidjson::Value& json) {
  base::Status status = dag::Node::deserialize(json);
  if (status != base::kStatusCodeOk) {
    return status;
  }
  if (json.HasMember("path_") && json["path_"].IsString()) {
    path_ = json["path_"].GetString();
  }

  return base::kStatusCodeOk;
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
  outputs_[0]->notifyWritten(prompt);
  index_++;
  return base::kStatusCodeOk;
}

base::Status PrintNode::run() {
  tokenizer::TokenizerText* result =
      (tokenizer::TokenizerText*)(inputs_[0]->getParam(this));
  if (result == nullptr) {
    return base::kStatusCodeErrorInvalidValue;
  }
  std::cout << "A: " << result->texts_[0].c_str() << std::endl;

  // char cwd[PATH_MAX];
  // if (!getcwd(cwd, sizeof(cwd))) {
  //   NNDEPLOY_LOGE("PrintNode] Failed to get current working directory\n");
  //   return base::kStatusCodeErrorIO;
  // }
  std::string cwd = base::getcwd();

  time_t t = time(NULL);
  std::ostringstream oss;
  oss << cwd.c_str() << "/.tmp";
  std::string tmp_dir = oss.str();

  // if (access(tmp_dir.c_str(), F_OK) != 0) {
  //   if (mkdir(tmp_dir.c_str(), 0755) != 0) {
  //     NNDEPLOY_LOGE("[PrintNode] Failed to create directory\n");
  //     return base::kStatusCodeErrorIO;
  //   }
  // }
  if (!base::isDirectory(tmp_dir)) {
    if (!base::createDirectory(tmp_dir)) {
      NNDEPLOY_LOGE("[PrintNode] Failed to create directory\n");
      return base::kStatusCodeErrorIO;
    }
  }

  // std::string file_path = tmp_dir + "/output.txt";
  if (path_.empty()) {
    path_ = tmp_dir + "/output_" + name_ + ".txt";
  }

  std::ofstream ofs(path_.c_str());
  if (!ofs) {
    NNDEPLOY_LOGE("[PrintNode] Failed to open file\n");
    return base::kStatusCodeErrorIO;
  }
  ofs << result->texts_[0];
  ofs.close();

  NNDEPLOY_LOGI("[PrintNode] Text written to: %s\n", path_.c_str());
  return base::kStatusCodeOk;
}

device::Tensor* PrefillEmbeddingNode::genPastKeyValue(
    const std::vector<int32_t>& kv_init_shape) {
  device::Device* device = device::getDefaultHostDevice();
  device::TensorDesc past_kv_desc;
  past_kv_desc.data_type_ = base::dataTypeOf<float>();
  past_kv_desc.data_format_ = base::DataFormat::kDataFormatS1D;
  past_kv_desc.shape_ = kv_init_shape;
  device::Tensor* past_kv;
  past_kv = new device::Tensor(device, past_kv_desc, "past_key_values");
  return past_kv;
}

device::Tensor* PrefillEmbeddingNode::genPositionIds(
    int seq_len, int all_seq_len, base::DataType data_type,
    base::DataFormat data_format) {
  device::Device* device = device::getDefaultHostDevice();
  device::TensorDesc position_ids_desc;
  position_ids_desc.data_type_ = data_type;
  position_ids_desc.data_format_ = data_format;
  position_ids_desc.shape_ = {1, seq_len};
  device::Tensor* position_ids;
  position_ids = new device::Tensor(device, position_ids_desc, "position_ids");

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
  attention_mask =
      new device::Tensor(device, attention_mask_desc, "attention_mask");

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
  inputs_embeds = new device::Tensor(device, input_id_desc, "input_ids");

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

  PrefillEmbeddingParam* embedding_param =
      (PrefillEmbeddingParam*)(param_.get());
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

  auto past_kv = genPastKeyValue(embedding_param->kv_init_shape_);
  NNDEPLOY_LOGI("prefill past_kv name: %s\n", past_kv->getName().c_str());
  past_kv->getDesc().print();

  outputs_[0]->set(inputs_embeds, false);
  outputs_[1]->set(attention_mask, false);
  outputs_[2]->set(position_ids, false);
  outputs_[3]->set(past_kv, false);

  return status;
}

base::Status PrefillSampleNode::run() {
  base::Status status = base::kStatusCodeOk;

  device::Tensor* logits = inputs_[0]->getTensor(this);
  tokenizer::TokenizerIds* token_ids =
      (tokenizer::TokenizerIds*)inputs_[1]->getParam(this);
  std::vector<int32_t> history_ids = token_ids->ids_[0];

  static int index = 0;
  if (index == 0) {
    std::string debug_file = "old_logits.csv";
    std::ofstream debug_file_stream(debug_file);
    logits->print(debug_file_stream);
    debug_file_stream.close();
    index++;
  }

  int32_t out_token_id = sample(logits, history_ids);
  tokenizer::TokenizerIds* out_token = new tokenizer::TokenizerIds();
  NNDEPLOY_LOGI("out_token_id: %d\n", out_token_id);

  out_token->ids_.push_back({out_token_id});

  outputs_[0]->set(out_token, false);
  outputs_[0]->notifyWritten(out_token);

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

// QwenPrefill

base::Status QwenPrefill::serialize(
    rapidjson::Value& json, rapidjson::Document::AllocatorType& allocator) {
  base::Status status = dag::CompositeNode::serialize(json, allocator);
  if (status != base::kStatusCodeOk) {
    return status;
  }
  rapidjson::Value config_path_val;
  config_path_val.SetString(
      config_path_.c_str(),
      static_cast<rapidjson::SizeType>(config_path_.length()), allocator);
  json.AddMember("config_path_", config_path_val, allocator);
  return base::kStatusCodeOk;
}
base::Status QwenPrefill::deserialize(rapidjson::Value& json) {
  base::Status status = dag::CompositeNode::deserialize(json);
  if (status != base::kStatusCodeOk) {
    return status;
  }
  if (json.HasMember("config_path_") && json["config_path_"].IsString()) {
    config_path_ = json["config_path_"].GetString();
  }
  return base::kStatusCodeOk;
}

base::Status QwenPrefill::init() {
  base::Status status = base::kStatusCodeOk;

  setConfigParam();

  status = prefill_token_node_->init();
  NNDEPLOY_RETURN_ON_NEQ(status, base::kStatusCodeOk,
                         "prefill_token_node_ init failed!");
  status = prefill_embedding_node_->init();
  NNDEPLOY_RETURN_ON_NEQ(status, base::kStatusCodeOk,
                         "prefill_embedding_node_ init failed!");
  status = prefill_infer_node_->init();
  NNDEPLOY_RETURN_ON_NEQ(status, base::kStatusCodeOk,
                         "prefill_infer_node_ init failed!");
  status = prefill_sample_node_->init();
  NNDEPLOY_RETURN_ON_NEQ(status, base::kStatusCodeOk,
                         "prefill_sample_node_ init failed!");

  return status;
}

base::Status QwenPrefill::run() {
  base::Status status = base::kStatusCodeOk;
  setRunningFlag(true);

  tokenizer::TokenizerText* prompt =
      (tokenizer::TokenizerText*)inputs_[0]->getParam(this);
  prefill_token_node_->getInput(0)->set(prompt, true);
  prefill_token_node_->updateInput();

  prefill_token_node_->run();
  prefill_embedding_node_->run();
  prefill_infer_node_->run();
  prefill_sample_node_->run();

  tokenizer::TokenizerIds* out_token =
      (tokenizer::TokenizerIds*)prefill_sample_node_->getOutput(0)->getParam(
          prefill_sample_node_);
  tokenizer::TokenizerIds* out_token_out = new tokenizer::TokenizerIds();
  *out_token_out = *out_token;
  outputs_[0]->set(out_token_out, false);
  outputs_[0]->notifyWritten(out_token_out);

  device::Tensor* presents =
      (device::Tensor*)prefill_infer_node_->getOutput(1)->getTensor(
          prefill_infer_node_);
  device::Device* device = device::getDefaultHostDevice();
  device::TensorDesc presents_desc = presents->getDesc();
  NNDEPLOY_LOGI("prefill presents name: %s\n", presents->getName().c_str());
  presents->getDesc().print();
  std::string debug_file = "old_prefill_presents.csv";
  std::ofstream debug_file_stream(debug_file);
  presents->print(debug_file_stream);
  debug_file_stream.close();
  device::Tensor* presents_out = outputs_[1]->create(device, presents_desc);
  presents->copyTo(presents_out);
  outputs_[1]->notifyWritten(presents_out);

  tokenizer::TokenizerIds* history_token =
      (tokenizer::TokenizerIds*)prefill_token_node_->getOutput(0)->getParam(
          prefill_token_node_);
  tokenizer::TokenizerIds* history_token_out = new tokenizer::TokenizerIds();
  *history_token_out = *history_token;
  outputs_[2]->set(history_token_out, false);
  outputs_[2]->notifyWritten(history_token_out);

  setRunningFlag(false);
  return status;
}

base::Status QwenPrefill::deinit() {
  base::Status status = base::kStatusCodeOk;
  status = prefill_token_node_->deinit();
  NNDEPLOY_RETURN_ON_NEQ(status, base::kStatusCodeOk,
                         "prefill_token_node_ deinit failed!");
  status = prefill_embedding_node_->deinit();
  NNDEPLOY_RETURN_ON_NEQ(status, base::kStatusCodeOk,
                         "prefill_embedding_node_ deinit failed!");
  status = prefill_infer_node_->deinit();
  NNDEPLOY_RETURN_ON_NEQ(status, base::kStatusCodeOk,
                         "prefill_infer_node_ deinit failed!");
  status = prefill_sample_node_->deinit();
  NNDEPLOY_RETURN_ON_NEQ(status, base::kStatusCodeOk,
                         "prefill_sample_node_ deinit failed!");
  return status;
}

base::Status QwenPrefill::setInferenceType(base::InferenceType inference_type) {
  prefill_infer_node_->setInferenceType(inference_type);
  return base::kStatusCodeOk;
}

base::Status QwenPrefill::defaultParam() {
  dag::NodeDesc token_desc("token_node", {"prefill_input"},
                           {"prefill_token_ids"});
  this->setNodeDesc(prefill_token_node_, token_desc);

  dag::NodeDesc embedding_desc(
      "embedding_node", {"prefill_token_ids"},
      {"prefill_input_ids", "prefill_attention_mask", "prefill_position_ids",
       "prefill_past_key_values"});
  this->setNodeDesc(prefill_embedding_node_, embedding_desc);

  dag::NodeDesc infer_desc("prefill_infer",
                           {"prefill_input_ids", "prefill_attention_mask",
                            "prefill_position_ids", "prefill_past_key_values"},
                           {"prefill_logits", "prefill_presents"});
  this->setNodeDesc(prefill_infer_node_, infer_desc);

  dag::NodeDesc sample_desc("prefill_sample_node",
                            {"prefill_logits", "prefill_token_ids"},
                            {"prefill_out_ids"});
  this->setNodeDesc(prefill_sample_node_, sample_desc);
  return base::kStatusCodeOk;
}

base::Status QwenPrefill::setConfigParam() {
  QwenConfig config = parseConfig(config_path_);
  PrefillEmbeddingParam* embedding_param =
      dynamic_cast<PrefillEmbeddingParam*>(prefill_embedding_node_->getParam());
  embedding_param->hidden_size_ = config.hidden_size_;
  embedding_param->embedding_file_ = config.embedding_file_;
  embedding_param->kv_init_shape_ = config.kv_init_shape_;

  tokenizer::TokenizerPraram* token_param =
      dynamic_cast<tokenizer::TokenizerPraram*>(
          prefill_token_node_->getParam());
  token_param->is_path_ = false;
  token_param->tokenizer_type_ =
      nndeploy::tokenizer::TokenizerType::kTokenizerTypeHF;
  token_param->json_blob_ = config.tokenizer_json_;

  inference::InferenceParam* inference_param =
      (inference::InferenceParam*)(prefill_infer_node_->getParam());
  inference_param->is_path_ = true;
  inference_param->model_value_ = {config.model_value_};

  return base::kStatusCodeOk;
}

base::Status QwenPrefill::setInferParams(bool is_path,
                                         base::ModelType model_type,
                                         base::DeviceType device_type) {
  inference::InferenceParam* inference_param =
      (inference::InferenceParam*)(prefill_infer_node_->getParam());
  inference_param->is_path_ = true;
  inference_param->device_type_ = device_type;
  inference_param->model_type_ = model_type;

  return base::kStatusCodeOk;
}

base::Status DecodeSampleNode::run() {
  base::Status status = base::kStatusCodeOk;

  device::Tensor* logits = inputs_[0]->getTensor(this);

  DecodeSampleParam* sample_params =
      dynamic_cast<DecodeSampleParam*>(param_.get());
  std::vector<int32_t> history_ids = sample_params->history_ids_.ids_[0];

  int32_t out_token_id = sample(logits, history_ids_);
  tokenizer::TokenizerIds* out_token = new tokenizer::TokenizerIds();

  /* when first time decode, append the prefill preditct token */
  if (is_first_) {
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

base::Status DecodeEmbeddingNode::run() {
  base::Status status = base::kStatusCodeOk;

  DecodeEmbeddingParam* embedding_param = (DecodeEmbeddingParam*)(param_.get());
  tokenizer::TokenizerIds* decode_token_ids =
      (tokenizer::TokenizerIds*)(inputs_[0]->getParam(this));
  std::vector<std::vector<int32_t>> token_ids = decode_token_ids->ids_;
  if (decode_token_ids->ids_[0].size() == 2) {
    token_ids[0] = {decode_token_ids->ids_[0][1]};
  } else {
    token_ids = decode_token_ids->ids_;
  }
  std::string embedding_file = embedding_param->embedding_file_;
  int hidden_size = embedding_param->hidden_size_;
  int seq_len = token_ids[0].size();
  int all_seq_len = embedding_param->all_seq_len_;
  past_kv_ = inputs_[1]->getTensor(this);
  NNDEPLOY_LOGI("decode past_kv name: %s\n", past_kv_->getName().c_str());
  past_kv_->getDesc().print();
  past_kv_->setName("past_key_values");

  auto inputs_embeds = genEmbedding(
      token_ids[0], seq_len, hidden_size, embedding_param->data_type_,
      embedding_param->data_format_, embedding_file);

  auto attention_mask =
      genAttentionMask(seq_len, all_seq_len, embedding_param->data_type_,
                       embedding_param->data_format_);
  NNDEPLOY_LOGI("decode attention_mask name: %s\n", attention_mask->getName().c_str());
  attention_mask->getDesc().print();

  auto position_ids =
      genPositionIds(seq_len, all_seq_len, embedding_param->posid_data_type_,
                     embedding_param->data_format_);
  NNDEPLOY_LOGI("decode position_ids name: %s\n", position_ids->getName().c_str());
  position_ids->getDesc().print();

  if (is_first_) is_first_ = false;
  outputs_[0]->set(inputs_embeds, false);
  outputs_[1]->set(attention_mask, false);
  outputs_[2]->set(position_ids, false);
  outputs_[3]->set(past_kv_, true);

  return status;
}

device::Tensor* DecodeEmbeddingNode::genPositionIds(
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

device::Tensor* DecodeEmbeddingNode::genAttentionMask(
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

device::Tensor* DecodeEmbeddingNode::genEmbedding(
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

// QwenDecode

base::Status QwenDecode::serialize(
    rapidjson::Value& json, rapidjson::Document::AllocatorType& allocator) {
  base::Status status = dag::CompositeNode::serialize(json, allocator);
  if (status != base::kStatusCodeOk) {
    return status;
  }
  rapidjson::Value config_path_val;
  config_path_val.SetString(
      config_path_.c_str(),
      static_cast<rapidjson::SizeType>(config_path_.length()), allocator);
  json.AddMember("config_path_", config_path_val, allocator);
  return base::kStatusCodeOk;
}
base::Status QwenDecode::deserialize(rapidjson::Value& json) {
  base::Status status = dag::CompositeNode::deserialize(json);
  if (status != base::kStatusCodeOk) {
    return status;
  }
  if (json.HasMember("config_path_") && json["config_path_"].IsString()) {
    config_path_ = json["config_path_"].GetString();
  }
  return base::kStatusCodeOk;
}

base::Status QwenDecode::init() {
  base::Status status = base::kStatusCodeOk;

  setConfigParam();

  decode_embedding_node_->init();
  NNDEPLOY_RETURN_ON_NEQ(status, base::kStatusCodeOk,
                         "decode_embedding_node_ init failed!");
  decode_infer_node_->init();
  NNDEPLOY_RETURN_ON_NEQ(status, base::kStatusCodeOk,
                         "decode_infer_node_ init failed!");
  decode_sample_node_->init();
  NNDEPLOY_RETURN_ON_NEQ(status, base::kStatusCodeOk,
                         "decode_sample_node_ init failed!");
  decode_node_->init();
  NNDEPLOY_RETURN_ON_NEQ(status, base::kStatusCodeOk,
                         "decode_node_ init failed!");

  return status;
}

base::Status QwenDecode::deinit() {
  base::Status status = base::kStatusCodeOk;

  decode_embedding_node_->deinit();
  NNDEPLOY_RETURN_ON_NEQ(status, base::kStatusCodeOk,
                         "decode_embedding_node_ deinit failed!");
  decode_infer_node_->deinit();
  NNDEPLOY_RETURN_ON_NEQ(status, base::kStatusCodeOk,
                         "decode_infer_node_ deinit failed!");
  decode_sample_node_->deinit();
  NNDEPLOY_RETURN_ON_NEQ(status, base::kStatusCodeOk,
                         "decode_sample_node_ deinit failed!");
  decode_node_->deinit();
  NNDEPLOY_RETURN_ON_NEQ(status, base::kStatusCodeOk,
                         "decode_node_ deinit failed!");

  return status;
}

void QwenDecode::getStopTokens(std::string& token_file) {
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

base::Status QwenDecode::run() {
  base::Status status = base::kStatusCodeOk;

  setRunningFlag(true);

  // tokenizer::TokenizerIds* prefill_token_ids = new tokenizer::TokenizerIds();
  // prefill_token_ids->ids_ =
  //     ((tokenizer::TokenizerIds*)(inputs_[0]->getParam(this)))->ids_;
  tokenizer::TokenizerIds* prefill_token_ids =
      ((tokenizer::TokenizerIds*)(inputs_[0]->getParam(this)));
  device::Tensor* present_kv = (device::Tensor*)(inputs_[1]->getTensor(this));
  tokenizer::TokenizerIds* history_ids =
      (tokenizer::TokenizerIds*)(inputs_[2]->getParam(this));

  history_ids->ids_[0].push_back(prefill_token_ids->ids_[0][0]);
  history_ids_ = *history_ids;

  decode_embedding_node_->getInput(0)->set(prefill_token_ids, true);
  decode_embedding_node_->getInput(1)->set(present_kv, true);
  // decode_embedding_node_->updateInput();

  DecodeEmbeddingParam* embedding_param =
      dynamic_cast<DecodeEmbeddingParam*>(decode_embedding_node_->getParam());
  DecodeSampleParam* sample_param =
      dynamic_cast<DecodeSampleParam*>(decode_sample_node_->getParam());
  embedding_param->all_seq_len_ = history_ids_.ids_[0].size();
  sample_param->history_ids_ = history_ids_;

  int iters = loops();
  for (int i = 0; i < iters; ++i) {
    if (isStop()) {
      tokenizer::TokenizerText* word =
          (tokenizer::TokenizerText*)(decode_node_->getOutput(0)->getParam(
              decode_node_));
      word->texts_[0] = result_;
      break;
    }

    embedding_param->gen_seq_len_++;

    decode_embedding_node_->run();
    decode_infer_node_->run();
    decode_sample_node_->run();
    decode_node_->run();

    embedding_param->all_seq_len_++;

    tokenizer::TokenizerIds* token_id =
        (tokenizer::TokenizerIds*)(decode_sample_node_->getOutput(0)->getParam(
            decode_sample_node_));
    decode_embedding_node_->getInput(0)->set(token_id, true);
    device::Tensor* past_kv =
        decode_infer_node_->getOutput(1)->getTensor(decode_infer_node_);
    NNDEPLOY_LOGI("decode past_kv name: %s\n", past_kv->getName().c_str());
    past_kv->getDesc().print();
    decode_embedding_node_->getInput(1)->set(past_kv, true);

    history_ids_.ids_ = sample_param->history_ids_.ids_;
    tokenizer::TokenizerText* word =
        (tokenizer::TokenizerText*)(decode_node_->getOutput(0)->getParam(
            decode_node_));
    int history_size = history_ids_.ids_[0].size();
    int out_token_id = history_ids_.ids_[0][history_size - 1];
    bool not_append =
        std::find(sample_param->stop_tokens_.ids_[0].begin(),
                  sample_param->stop_tokens_.ids_[0].end(),
                  out_token_id) != sample_param->stop_tokens_.ids_[0].end();
    if (!not_append) result_.append(word->texts_[0]);
  }

  tokenizer::TokenizerText* word =
      (tokenizer::TokenizerText*)(decode_node_->getOutput(0)->getParam(
          decode_node_));
  outputs_[0]->set(word);
  outputs_[0]->notifyWritten(word);

  setRunningFlag(false);

  return status;
}

base::Status QwenDecode::setInferenceType(base::InferenceType inference_type) {
  decode_infer_node_->setInferenceType(inference_type);
  return base::kStatusCodeOk;
}

base::Status QwenDecode::defaultParam() {
  dag::NodeDesc embedding_desc(
      "embedding_node", {"decode_prefill_token_ids", "decode_prefill_presents"},
      {"decode_input_ids", "decode_attention_mask", "decode_position_ids",
       "decode_past_key_values"});
  this->setNodeDesc(decode_embedding_node_, embedding_desc);

  dag::NodeDesc infer_desc("decode_infer",
                           {"decode_input_ids", "decode_attention_mask",
                            "decode_position_ids", "decode_past_key_values"},
                           {"decode_logits", "decode_presents"});
  this->setNodeDesc(decode_infer_node_, infer_desc);

  dag::NodeDesc sample_desc("sample_node", {"decode_logits"},
                            {"decode_out_ids"});
  this->setNodeDesc(decode_sample_node_, sample_desc);

  dag::NodeDesc decode_desc("decode_node", {"decode_out_ids"}, {"decode_out"});
  this->setNodeDesc(decode_node_, decode_desc);

  return base::kStatusCodeOk;
}

base::Status QwenDecode::setConfigParam() {
  QwenConfig config = parseConfig(config_path_);
  max_seq_len_ = config.max_seq_len_;
  getStopTokens(config.tokenizer_txt_);

  DecodeEmbeddingParam* embedding_param =
      dynamic_cast<DecodeEmbeddingParam*>(decode_embedding_node_->getParam());
  embedding_param->embedding_file_ = config.embedding_file_;
  embedding_param->hidden_size_ = config.hidden_size_;
  embedding_param->gen_seq_len_ = 1;

  DecodeSampleParam* sample_params =
      dynamic_cast<DecodeSampleParam*>(decode_sample_node_->getParam());
  sample_params->stop_tokens_.ids_.push_back(stop_tokens_);

  tokenizer::TokenizerPraram* token_param =
      dynamic_cast<tokenizer::TokenizerPraram*>(decode_node_->getParam());
  token_param->is_path_ = false;
  token_param->tokenizer_type_ =
      nndeploy::tokenizer::TokenizerType::kTokenizerTypeHF;
  token_param->json_blob_ = config.tokenizer_json_;

  inference::InferenceParam* inference_param =
      (inference::InferenceParam*)(decode_infer_node_->getParam());
  inference_param->model_value_ = {config.model_value_};

  return base::kStatusCodeOk;
}

base::Status QwenDecode::setInferParams(bool is_path,
                                        base::ModelType model_type,
                                        base::DeviceType device_type) {
  inference::InferenceParam* inference_param =
      (inference::InferenceParam*)(decode_infer_node_->getParam());
  inference_param->is_path_ = true;
  inference_param->device_type_ = device_type;
  inference_param->model_type_ = model_type;

  return base::kStatusCodeOk;
}

dag::Graph* createQwenGraph(const std::string& name,
                            base::InferenceType inference_type,
                            base::DeviceType device_type, dag::Edge* prompt,
                            dag::Edge* out, base::ModelType model_type,
                            bool is_path,
                            std::vector<std::string> config_path) {
  dag::Graph* llm_graph = new dag::Graph(name, {prompt}, {out});

  dag::Edge* prefill_out_ids = llm_graph->createEdge("out_ids");
  dag::Edge* prefill_presents = llm_graph->createEdge("presents");
  dag::Edge* history_ids = llm_graph->createEdge("history_ids");

  std::vector<dag::Edge*> prefill_in = {prompt};
  std::vector<dag::Edge*> prefill_out = {prefill_out_ids, prefill_presents,
                                         history_ids};
  QwenPrefill* prefill = new QwenPrefill("prefill", prefill_in, prefill_out);
  prefill->setConfigPath(config_path[0]);
  prefill->setInferenceType(inference_type);
  prefill->defaultParam();
  prefill->setInferParams(is_path, model_type, device_type);
  llm_graph->addNode(prefill, false);

  std::vector<dag::Edge*> decode_out = {out};
  QwenDecode* decode = new QwenDecode("decode", prefill_out, decode_out);
  decode->setConfigPath(config_path[0]);
  decode->setInferenceType(inference_type);
  decode->defaultParam();
  decode->setInferParams(is_path, model_type, device_type);
  llm_graph->addNode(decode, false);

  return llm_graph;
}

REGISTER_NODE("nndeploy::qwen::PrefillEmbeddingNode", PrefillEmbeddingNode);
REGISTER_NODE("nndeploy::qwen::DecodeEmbeddingNode", DecodeEmbeddingNode);
REGISTER_NODE("nndeploy::qwen::PrefillSampleNode", PrefillSampleNode);
REGISTER_NODE("nndeploy::qwen::DecodeSampleNode", DecodeSampleNode);
REGISTER_NODE("nndeploy::qwen::PromptNode", PromptNode);
REGISTER_NODE("nndeploy::qwen::PrintNode", PrintNode);
REGISTER_NODE("nndeploy::qwen::QwenPrefill", QwenPrefill);
REGISTER_NODE("nndeploy::qwen::QwenDecode", QwenDecode);

}  // namespace qwen
}  // namespace nndeploy
