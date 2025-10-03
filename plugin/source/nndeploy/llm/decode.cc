#include "nndeploy/llm/decode.h"

#include "nndeploy/tokenizer/tokenizer_cpp/tokenizer_cpp.h"

namespace nndeploy {
namespace llm {

Decode::Decode(const std::string& name, std::vector<dag::Edge*> inputs,
               std::vector<dag::Edge*> outputs)
    : dag::Loop(name, inputs, outputs) {
  key_ = "nndeploy::llm::Decode";
  desc_ = "Decode: Decode pipeline";

  this->setInputTypeInfo<tokenizer::TokenizerIds>("input_tokens");
  this->setOutputTypeInfo<tokenizer::TokenizerText>("output_text");

  decode_infer_node_ = dynamic_cast<llm::LlmInfer*>(
      this->createNode<llm::LlmInfer>("decode_infer"));
  decode_sampler_node_ = this->createNode<Sampler>("decode_sampler_node");
  decode_token_node_ =
      this->createNode<tokenizer::TokenizerDecodeCpp>("token_node");
  stream_out_node_ =
      dynamic_cast<StreamOut*>(this->createNode<StreamOut>("stream_out_node"));
}

Decode::~Decode() {}

base::Status Decode::make(const dag::NodeDesc& infer,
                          const dag::NodeDesc& sample,
                          const dag::NodeDesc& tokenizer,
                          const dag::NodeDesc& stream_out) {
  base::Status status = base::kStatusCodeOk;
  status = this->setNodeDesc(decode_infer_node_, infer);
  NNDEPLOY_RETURN_ON_NEQ(status, base::kStatusCodeOk,
                         "decode_infer_node_ set node desc failed!");
  status = this->setNodeDesc(decode_sampler_node_, sample);
  NNDEPLOY_RETURN_ON_NEQ(status, base::kStatusCodeOk,
                         "decode_sampler_node_ set node desc failed!");
  status = this->setNodeDesc(decode_token_node_, tokenizer);
  NNDEPLOY_RETURN_ON_NEQ(status, base::kStatusCodeOk,
                         "decode_token_node_ set node desc failed!");
  status = this->setNodeDesc(stream_out_node_, stream_out);
  NNDEPLOY_RETURN_ON_NEQ(status, base::kStatusCodeOk,
                         "stream_out_node_ set node desc failed!");
  return status;
}

base::Status Decode::initEnd() {
  max_seq_len_ = decode_infer_node_->getMaxSeqLen();
  if (!tokenizer_txt_.empty()) {
    this->getStopTokens(tokenizer_txt_);
  }
  return base::kStatusCodeOk;
}

base::Status Decode::iterAfter() {
  auto decode_sampler_node_output_ = decode_sampler_node_->getOutput(0);
  decode_infer_node_->setIterInput(decode_sampler_node_output_, 1);
  return base::kStatusCodeOk;
}

std::vector<dag::Edge*> Decode::forward(dag::Edge* input) {
  std::vector<dag::Edge*> output;
  for (int i = 0; i < this->loops(); i++) {
    output = (*decode_infer_node_)(input);
    output = (*decode_sampler_node_)(output);
    decode_infer_node_->setIterInput(output[0], 1);
    output.push_back(input);
    output = (*decode_token_node_)(output);
    output = (*stream_out_node_)(output);
  }
  return {output[0]};
}

void Decode::getStopTokens(std::string& token_file) {
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

int Decode::loops() {
  // 
  if (inputs_[0]->empty()) {
    NNDEPLOY_LOGE("Decode loops is 1\n");
    return 1;
  }
  if (isStop()) {
    NNDEPLOY_LOGE("Decode loops is -1\n");
    return -1;
  } else {
    NNDEPLOY_LOGE("Decode loops is %d\n", max_seq_len_);
    return max_seq_len_;
  }
}

bool Decode::isStop() {
  if (!stop_tokens_.empty()) {
    return this->isStopTokens();
  } else {
    return this->isStopTexts();
  }
}

bool Decode::isStopTokens() {
  tokenizer::TokenizerIds* token_ids;
  if (is_first_) {
    token_ids =
        (tokenizer::TokenizerIds*)(decode_infer_node_->getInput(0)->getParam(
            decode_infer_node_));
  } else {
    token_ids =
        (tokenizer::TokenizerIds*)(decode_token_node_->getInput(0)->getParam(
            decode_token_node_));
  }

  int token = token_ids->ids_[0][0];
  return std::find(stop_tokens_.begin(), stop_tokens_.end(), token) !=
         stop_tokens_.end();
}

bool Decode::isStopTexts() {
  tokenizer::TokenizerText* token_text;
  if (is_first_) {
    return false;
  } else {
    token_text =
        (tokenizer::TokenizerText*)(stream_out_node_->getInput(0)->getParam(
            stream_out_node_));
  }

  // TODO，大语言模型的输出字符是什么？
  std::string text = token_text->texts_[0];
  std::string last_text = "";
  size_t last_space = text.find_last_of(' ');
  if (last_space != std::string::npos) {
    last_text = text.substr(last_space + 1);
  } else {
    last_text = text;
  }
  return std::find(stop_texts_.begin(), stop_texts_.end(), last_text) !=
         stop_texts_.end();
}

base::Status Decode::serialize(rapidjson::Value& json,
                               rapidjson::Document::AllocatorType& allocator) {
  base::Status status = dag::Loop::serialize(json, allocator);
  if (status != base::kStatusCodeOk) {
    NNDEPLOY_LOGE("serialize decode failed\n");
    return status;
  }
  json.AddMember("tokenizer_txt_",
                 rapidjson::Value(tokenizer_txt_.c_str(), allocator),
                 allocator);
  return status;
}

base::Status Decode::deserialize(rapidjson::Value& json) {
  base::Status status = dag::Loop::deserialize(json);
  if (status != base::kStatusCodeOk) {
    return status;
  }
  if (json.HasMember("tokenizer_txt_") && json["tokenizer_txt_"].IsString()) {
    tokenizer_txt_ = json["tokenizer_txt_"].GetString();
  }
  return status;
}

REGISTER_NODE("nndeploy::llm::Decode", Decode);

}  // namespace llm
}  // namespace nndeploy
