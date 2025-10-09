//
//  sample.cpp
//
//  Created by MNN on 2023/09/25.
//  ZhaodeWang
//

#include "nndeploy/llm/sample.h"

#include "nndeploy/op/op.h"
#include "nndeploy/op/op_mul.h"
#include "nndeploy/op/op_softmax.h"

namespace nndeploy {
namespace llm {

// sampler compute struct start
// a index and its corresponding score
struct IndexScore {
  int index;
  float score;
};
struct IndexScoreCmpLess {
  bool operator()(IndexScore a, IndexScore b) { return a.score < b.score; }
};
struct IndexScoreCmpGreater {
  bool operator()(IndexScore a, IndexScore b) { return a.score > b.score; }
};
// a series of index and their corresponding logits
struct SubsetLogits {
  SubsetLogits()
      : logits(nullptr), index(std::vector<int>()), is_subset(false) {}
  SubsetLogits(device::Tensor* logits, std::vector<int> index, bool is_subset)
      : logits(logits), index(index), is_subset(is_subset) {}
  ~SubsetLogits() {
    if (logits != nullptr && is_external == false) {
      delete logits;
    }
  }

  // 拷贝构造函数
  SubsetLogits(const SubsetLogits& other)
      : index(other.index),
        is_external(true),
        logits(other.logits),
        is_subset(other.is_subset) {}

  // 赋值运算符重载
  SubsetLogits& operator=(const SubsetLogits& other) {
    if (this != &other) {
      index = other.index;
      logits = other.logits;
      is_external = true;
      is_subset = other.is_subset;
    }
    return *this;
  }

  std::vector<int> index;
  bool is_external = true;
  device::Tensor* logits;
  bool is_subset;
};
// sampler compute struct end

// sampler compute functions start
device::Tensor* tempratureSoftmax(device::Tensor* logits, float temperature,
                                  int axis = -1) {
  // create temperature tensor
  base::DataType data_type = logits->getDataType();
  base::DataFormat data_format = logits->getDataFormat();
  base::IntVector shape = logits->getShape();
  for (int i = 0; i < shape.size(); i++) {
    shape[i] = 1;
  }
  device::TensorDesc temperature_tensor_desc(data_type, data_format, shape);
  device::Tensor* temperature_tensor = new device::Tensor(
      logits->getDevice(), temperature_tensor_desc, "temperature");
  temperature_tensor->set(1.0f / temperature);
  device::Tensor* result = new device::Tensor("temprature_softmax.output");
  op::mul(logits, temperature_tensor, result);
  delete temperature_tensor;

  // create softmax param
  std::shared_ptr<ir::SoftmaxParam> param =
      std::make_shared<ir::SoftmaxParam>();
  param->axis_ = axis;
  op::softmax(result, param, result);

  return result;
}

SubsetLogits createSubsetLogits(device::Tensor* logits) {
  struct SubsetLogits subset;
  subset.logits = logits;
  subset.is_subset = false;
  return subset;
}

SubsetLogits createSubsetLogits(device::Tensor* logits,
                                const std::vector<int>& index) {
  struct SubsetLogits subset;
  subset.logits = logits;
  subset.index = index;
  subset.is_subset = true;
  return subset;
}

SubsetLogits createSubsetLogits(int size) {
  struct SubsetLogits subset;
  // TODO,暂时不能确定具体的类型
  base::DataType data_type = base::dataTypeOf<int>();
  base::DataFormat data_format = base::kDataFormatNC;
  base::IntVector shape = {1, size};
  device::TensorDesc logits_tensor_desc(data_type, data_format, shape);
  subset.logits = new device::Tensor(logits_tensor_desc);
  subset.is_external = false;
  subset.index.resize(size);
  subset.is_subset = true;
  return subset;
}

SubsetLogits createSubsetLogits(const std::vector<float>& scores,
                                const std::vector<int>& index) {
  int size = (int)(index.size());
  struct SubsetLogits subset;

  // TODO,暂时不能确定具体的类型
  base::DataType data_type = base::dataTypeOf<int>();
  base::DataFormat data_format = base::kDataFormatNC;
  base::IntVector shape = {1, size};
  device::TensorDesc logits_tensor_desc(data_type, data_format, shape);
  subset.logits = new device::Tensor(logits_tensor_desc);
  subset.is_external = false;
  auto pointer = (float*)(subset.logits->getData());
  for (int i = 0; i < size; ++i) {
    pointer[i] = scores[i];
  }

  subset.index = index;
  subset.is_subset = true;
  return subset;
}

void transformIndex(struct SubsetLogits& superset,
                    struct SubsetLogits& subset) {
  if (!(superset.is_subset)) return;
  for (auto& id : subset.index) {
    id = superset.index[id];
  }
}

int select(struct SubsetLogits& subset, int id) {
  if (!(subset.is_subset)) {
    return id;
  }
  return subset.index[id];
}

int argmaxSelect(struct SubsetLogits superset) {
  auto scores = (float*)(superset.logits->getData());
  // get last dimension index
  int lastIndex = superset.logits->getShape().size() - 1;
  // argmax size is last dimension size
  auto size = superset.logits->getShape()[lastIndex];
  float max_score = scores[0];
  int token_id = 0;
  for (int i = 0; i < size; i++) {
    float score = scores[i];
    if (score > max_score) {
      max_score = score;
      token_id = i;
    }
  }
  return select(superset, token_id);
}

int randomSelect(float* probs, size_t size) {
  std::random_device rd;
  std::mt19937 generator(rd());
  std::uniform_real_distribution<float> distribution(0.0, 1.0);
  float target = distribution(generator);
  float cumulative = 0.0;
  for (int i = 0; i < size; i++) {
    cumulative += probs[i];
    if (target < cumulative) {
      return i;
    }
  }
  return size - 1;
}

int randomSelect(device::Tensor* probs) {
  return randomSelect((float*)(probs->getData()), probs->getSize());
}

int reSoftmaxSelect(struct SubsetLogits subset, float temperature) {
  int token_index_id =
      randomSelect(tempratureSoftmax(subset.logits, temperature));
  return ((subset.is_subset) ? subset.index[token_index_id] : token_index_id);
}

int packSoftmax(device::Tensor* logits, std::vector<IndexScore>& index_scores,
                float temperature) {
  auto prob_varp = tempratureSoftmax(logits, temperature);
  auto probs = (float*)(prob_varp->getData());
  auto size = prob_varp->getSize();
  index_scores.resize(size);
  for (int i = 0; i < size; i++) {
    IndexScore m;
    m.index = i;
    m.score = probs[i];
    index_scores[i] = m;
  }
  return size;
}
// sampler compute functions end

// SampleParam
base::Status SampleParam::serialize(
    rapidjson::Value& json, rapidjson::Document::AllocatorType& allocator) {
  this->addDropdownParam(
      "sampler", {"greedy", "temperature", "topK", "topP", "minP", "tfs",
                  "typical", "penalty", "ngram"});
  base::Status status = base::Param::serialize(json, allocator);
  if (status != base::kStatusCodeOk) {
    NNDEPLOY_LOGE("SampleParam::serialize failed\n");
    return status;
  }

  json.AddMember("sampler", rapidjson::Value(sampler.c_str(), allocator),
                 allocator);

  json.AddMember("temperature", temperature, allocator);
  json.AddMember("topK", topK, allocator);
  json.AddMember("topP", topP, allocator);
  json.AddMember("minP", minP, allocator);
  json.AddMember("tfsZ", tfsZ, allocator);
  json.AddMember("typical", typical, allocator);
  json.AddMember("penalty", penalty, allocator);
  json.AddMember("ngram", ngram, allocator);
  json.AddMember("ngram_factor", ngram_factor, allocator);
  json.AddMember("max_penalty", max_penalty, allocator);

  rapidjson::Value mixed_samplers_array(rapidjson::kArrayType);
  for (const auto& sampler : mixed_samplers) {
    mixed_samplers_array.PushBack(rapidjson::Value(sampler.c_str(), allocator),
                                  allocator);
  }
  json.AddMember("mixed_samplers", mixed_samplers_array, allocator);

  return base::kStatusCodeOk;
}

base::Status SampleParam::deserialize(rapidjson::Value& json) {
  base::Status status = base::Param::deserialize(json);
  if (status != base::kStatusCodeOk) {
    NNDEPLOY_LOGE("SampleParam::deserialize failed\n");
    return status;
  }

  if (json.HasMember("sampler") && json["sampler"].IsString()) {
    sampler = json["sampler"].GetString();
  }
  if (json.HasMember("temperature") && json["temperature"].IsFloat()) {
    temperature = json["temperature"].GetFloat();
  }
  if (json.HasMember("topK") && json["topK"].IsInt()) {
    topK = json["topK"].GetInt();
  }
  if (json.HasMember("topP") && json["topP"].IsFloat()) {
    topP = json["topP"].GetFloat();
  }
  if (json.HasMember("minP") && json["minP"].IsFloat()) {
    minP = json["minP"].GetFloat();
  }
  if (json.HasMember("tfsZ") && json["tfsZ"].IsFloat()) {
    tfsZ = json["tfsZ"].GetFloat();
  }
  if (json.HasMember("typical") && json["typical"].IsFloat()) {
    typical = json["typical"].GetFloat();
  }
  if (json.HasMember("penalty") && json["penalty"].IsFloat()) {
    penalty = json["penalty"].GetFloat();
  }
  if (json.HasMember("ngram") && json["ngram"].IsInt()) {
    ngram = json["ngram"].GetInt();
  }
  if (json.HasMember("ngram_factor") && json["ngram_factor"].IsFloat()) {
    ngram_factor = json["ngram_factor"].GetFloat();
  }
  if (json.HasMember("max_penalty") && json["max_penalty"].IsFloat()) {
    max_penalty = json["max_penalty"].GetFloat();
  }
  if (json.HasMember("mixed_samplers") && json["mixed_samplers"].IsArray()) {
    mixed_samplers.clear();
    const auto& array = json["mixed_samplers"];
    for (rapidjson::SizeType i = 0; i < array.Size(); i++) {
      if (array[i].IsString()) {
        mixed_samplers.push_back(array[i].GetString());
      }
    }
  }

  return base::kStatusCodeOk;
}

// Sample
Sampler::Sampler(const std::string& name, std::vector<dag::Edge*> inputs,
                 std::vector<dag::Edge*> outputs)
    : dag::Node(name, inputs, outputs) {
  key_ = "nndeploy::llm::Sampler";
  desc_ =
      "Sample generates next token from model logits using various sampling "
      "strategies:\n"
      "1. Greedy sampling - select token with highest probability\n"
      "2. Temperature sampling - sample from temperature-scaled distribution\n"
      "3. Top-K sampling - sample from top K most likely tokens\n"
      "4. Top-P (nucleus) sampling - sample from tokens with cumulative "
      "probability <= P\n"
      "5. Min-P sampling - filter tokens below minimum probability threshold\n"
      "6. Repetition penalty - penalize repeated tokens/n-grams\n"
      "\n"
      "Inputs:\n"
      "- inputs[0]: Tensor containing model logits for next token prediction\n"
      "Outputs:\n"
      "- outputs[0]: TokenizerIds containing sampled token ID\n";
  param_ = std::make_shared<SampleParam>();
  this->setInputTypeInfo<device::Tensor>("logits");
  this->setOutputTypeInfo<tokenizer::TokenizerIds>("sampled_token");
}

Sampler::~Sampler() {}

int32_t Sampler::sampleOld(device::Tensor* logits) {
  std::vector<int>* history_ids =
      this->getResourceWithState<std::vector<int>>("history_tokens");
  std::unordered_set<int> ids_set(history_ids->begin(), history_ids->end());
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

base::Status Sampler::run() {
  auto logits = inputs_[0]->get<device::Tensor>(this);
  int batch_size = logits->getShape()[0];

  tokenizer::TokenizerIds* out_token = new tokenizer::TokenizerIds();
  out_token->ids_.resize(batch_size);

  static int index = 0;
  if (index == 0) {
    std::string debug_file = "new_logits.csv";
    std::ofstream debug_file_stream(debug_file);
    logits->print(debug_file_stream);
    debug_file_stream.close();
    index++;
  }

  // auto sampled_token_id = sample(logits);
  auto sampled_token_id = sampleOld(logits);
  out_token->ids_[0].push_back(sampled_token_id);
  NNDEPLOY_LOGI("sampled_token_id: %d\n", sampled_token_id);

  outputs_[0]->set(out_token, false);
  outputs_[0]->notifyWritten(out_token);
  return base::kStatusCodeOk;
}

struct SubsetLogits Sampler::topK(struct SubsetLogits superset) {
  int K = (dynamic_cast<SampleParam*>(param_.get()))->topK;
  auto scores = (float*)(superset.logits->getData());
  auto size = superset.logits->getSize();
  // 1. time complexity: O(nlogk)
  std::priority_queue<IndexScore, std::vector<IndexScore>, IndexScoreCmpGreater>
      heap;
  for (int i = 0; i < size; i++) {
    IndexScore m;
    m.index = i;
    m.score = scores[i];
    if (heap.size() < K) {
      heap.push(m);
    } else {
      if (heap.top().score < m.score) {
        heap.pop();
        heap.push(m);
      }
    }
  }
  // 2. store top K results
  auto subset = createSubsetLogits(K);
  float* topKscores = (float*)(subset.logits->getData());
  for (int i = 0; i < K; i++) {
    subset.index[K - i - 1] = heap.top().index;
    topKscores[K - i - 1] = heap.top().score;
    heap.pop();
  }
  transformIndex(superset, subset);
  return subset;
}

struct SubsetLogits Sampler::topP(struct SubsetLogits superset) {
  float p = (dynamic_cast<SampleParam*>(param_.get()))->topP,
        temperature = (dynamic_cast<SampleParam*>(param_.get()))->temperature;
  std::vector<IndexScore> index_scores;
  int size = packSoftmax(superset.logits, index_scores, temperature);
  // 1. make max heap
  std::make_heap(index_scores.begin(), index_scores.end(), IndexScoreCmpLess());
  // 2. top p algorithm
  auto scores = (float*)(superset.logits->getData());
  std::vector<int> index;
  std::vector<float> subset_logits;
  float cumulative = 0.0f;
  while (cumulative < p && !index_scores.empty()) {
    std::pop_heap(index_scores.begin(), index_scores.end(),
                  IndexScoreCmpLess());
    IndexScore m = index_scores.back();
    index_scores.pop_back();
    index.push_back(m.index);
    subset_logits.push_back(scores[m.index]);
    cumulative += m.score;
  }
  auto subset = createSubsetLogits(subset_logits, index);
  transformIndex(superset, subset);
  return subset;
}

struct SubsetLogits Sampler::minP(struct SubsetLogits superset) {
  float p = (dynamic_cast<SampleParam*>(param_.get()))->minP,
        temperature = (dynamic_cast<SampleParam*>(param_.get()))->temperature;
  std::vector<IndexScore> index_scores;
  int size = packSoftmax(superset.logits, index_scores, temperature);
  // 1. make max heap
  std::make_heap(index_scores.begin(), index_scores.end(), IndexScoreCmpLess());
  // 2. min p algorithm
  auto scores = (float*)(superset.logits->getData());
  std::vector<int> index;
  std::vector<float> subset_logits;
  for (int i = 0; i < size; ++i) {
    std::pop_heap(index_scores.begin(), index_scores.end(),
                  IndexScoreCmpLess());
    IndexScore m = index_scores.back();
    if (m.score < p && !index.empty()) break;
    index_scores.pop_back();
    index.push_back(m.index);
    subset_logits.push_back(scores[m.index]);
  }
  auto subset = createSubsetLogits(subset_logits, index);
  transformIndex(superset, subset);
  return subset;
}

struct SubsetLogits Sampler::tfs(struct SubsetLogits superset) {
  float z = (dynamic_cast<SampleParam*>(param_.get()))->tfsZ,
        temperature = (dynamic_cast<SampleParam*>(param_.get()))->temperature;
  // tfs algorithm
  // 1. softmax
  std::vector<IndexScore> index_scores;
  int size = packSoftmax(superset.logits, index_scores, temperature);
  // 2. sort
  std::sort(index_scores.begin(), index_scores.end(), IndexScoreCmpGreater());
  auto scores = (float*)(superset.logits->getData());
  // 3. calculate derivatives
  std::vector<float> derivatives(size - 2, 0.0f);
  float first = index_scores[0].score - index_scores[1].score;
  float second = index_scores[1].score - index_scores[2].score;
  for (int i = 0; i < size - 2; ++i) {
    second = index_scores[i + 1].score - index_scores[i + 2].score;
    derivatives[i] = std::fabs(first - second);
    first = second;
  }
  // 4. normalize derivatives
  float derivatives_sum = 0.0;
  for (int i = 0; i < size - 2; ++i) derivatives_sum += derivatives[i];
  float derivatives_sum_rec = 1.0f / derivatives_sum;
  for (int i = 0; i < size - 2; ++i) derivatives[i] *= derivatives_sum_rec;
  // 5. cumulate, discard last 2 for sure.
  float cumulative = 0.0;
  std::vector<int> index;
  std::vector<float> subset_logits;
  for (int i = 0; i < size - 2; ++i) {
    IndexScore m = index_scores[i];
    cumulative += derivatives[i];
    if (cumulative >= z && !index.empty()) break;
    index.push_back(m.index);
    subset_logits.push_back(scores[m.index]);
  }
  auto subset = createSubsetLogits(subset_logits, index);
  transformIndex(superset, subset);
  return subset;
}

struct SubsetLogits Sampler::typical(struct SubsetLogits superset) {
  float p = (dynamic_cast<SampleParam*>(param_.get()))->typical,
        temperature = (dynamic_cast<SampleParam*>(param_.get()))->temperature;
  auto prob_varp = tempratureSoftmax(superset.logits, temperature);
  auto probs = (float*)(prob_varp->getData());
  auto size = prob_varp->getSize();
  std::vector<IndexScore> index_scores;
  index_scores.resize(size);
  // 1. calcaluate dist
  float entropy = 0.0f;
  for (int i = 0; i < size; i++) entropy -= probs[i] * std::log(probs[i]);
  for (int i = 0; i < size; i++) {
    IndexScore m;
    m.index = i;
    m.score = std::fabs(entropy + std::log(probs[i]));
    index_scores[i] = m;
  }
  // 2. make min heap for dist
  std::make_heap(index_scores.begin(), index_scores.end(),
                 IndexScoreCmpGreater());
  // 3. typical p algorithm
  auto scores = (float*)(superset.logits->getData());
  float cumulative = 0.0f;
  std::vector<int> index;
  std::vector<float> subset_logits;
  for (int i = 0; i < size; ++i) {
    std::pop_heap(index_scores.begin(), index_scores.end(),
                  IndexScoreCmpGreater());
    IndexScore m = index_scores.back();
    cumulative += probs[m.index];
    if (cumulative >= p && !index.empty()) break;
    index_scores.pop_back();
    index.push_back(m.index);
    subset_logits.push_back(scores[m.index]);
  }
  auto subset = createSubsetLogits(subset_logits, index);
  transformIndex(superset, subset);
  return subset;
}

// presence penalty
// no frequency penalty now!
struct SubsetLogits Sampler::penalty(struct SubsetLogits subset) {
  float penalty = (dynamic_cast<SampleParam*>(param_.get()))->penalty;
  int ngram = (dynamic_cast<SampleParam*>(param_.get()))->ngram;
  float ngram_factor = (dynamic_cast<SampleParam*>(param_.get()))->ngram_factor;
  float temperature = (dynamic_cast<SampleParam*>(param_.get()))->temperature;
  bool penalizeNgram = (ngram_factor > 1.0f);
  if (penalty <= 1.0f) return subset;  // no penalty!
  penalty = std::min(penalty,
                     (dynamic_cast<SampleParam*>(param_.get()))->max_penalty);
  // initialization
  std::vector<int>& prev =
      *(this->getResourceWithState<std::vector<int>>("history_tokens"));
  std::unordered_map<int, float> penalty_map;
  // 1. local ngram info, reversed order
  std::vector<int> ngram_info(ngram - 1);
  if (penalizeNgram) {
    for (int n = 0; n < ngram_info.size(); ++n) {
      ngram_info[n] = prev[prev.size() - 1 - n];
    }
  }
  // 2. generate penalty map
  for (int i = 0; i < prev.size(); ++i) {
    if (penalty_map.count(prev[i]) == 0) penalty_map[prev[i]] = penalty;
    if (penalizeNgram) {
      float ngram_penalty = penalty;
      for (int j = i - 1; i - j < ngram && j >= 0; --j) {
        int idx = i - j - 1;
        if (prev[j] != ngram_info[idx]) break;
        ngram_penalty *= ngram_factor;
        // no repeat larger than ngram!
        if (idx == ngram_info.size() - 1)
          ngram_penalty =
              (dynamic_cast<SampleParam*>(param_.get()))->max_penalty;
      }
      if (ngram_penalty > penalty_map[prev[i]])
        penalty_map[prev[i]] = ngram_penalty;
    }
  }
  // 3. penalize logits according to penalty_map
  auto scoresMap = (float*)(subset.logits->getData());
  for (auto it = penalty_map.begin(); it != penalty_map.end(); ++it) {
    scoresMap[it->first] = (scoresMap[it->first] >= 0.0f)
                               ? (scoresMap[it->first] / it->second)
                               : (scoresMap[it->first] * it->second);
  }
  return subset;
}

struct SubsetLogits Sampler::mixed(struct SubsetLogits subset) {
  for (auto sampler :
       (dynamic_cast<SampleParam*>(param_.get()))->mixed_samplers) {
    subset = subsetSampler(sampler, subset);
  }
  return subset;
}

struct SubsetLogits Sampler::subsetSampler(std::string sampler_type,
                                           struct SubsetLogits subset) {
  // 根据采样器类型执行相应的采样策略，对logits进行过滤和调整

  if (sampler_type == "penalty") {
    // 重复惩罚采样器：对历史出现过的token进行惩罚，降低其概率
    // 通过penalty参数控制惩罚强度，支持n-gram级别的惩罚机制
    subset = penalty(subset);
  }

  if (sampler_type == "topK") {
    // Top-K采样器：只保留概率最高的K个token，过滤掉其余token
    // 通过topK参数控制保留的token数量，减少采样空间
    subset = topK(subset);
  }

  if (sampler_type == "topP") {
    // Top-P (Nucleus)采样器：保留累积概率达到P的最小token集合
    // 通过topP参数控制累积概率阈值，动态调整采样空间大小
    subset = topP(subset);
  }

  if (sampler_type == "minP") {
    // Min-P采样器：过滤掉概率低于最高概率*minP倍数的token
    // 通过minP参数设置相对概率阈值，保持与最高概率的相对关系
    subset = minP(subset);
  }

  if (sampler_type == "tfs") {
    // Tail Free Sampling采样器：基于概率分布的尾部特征进行过滤
    // 通过tfsZ参数控制尾部截断程度，去除低质量的长尾token
    subset = tfs(subset);
  }

  if (sampler_type == "typical") {
    // Typical采样器：保留"典型"概率范围内的token，过滤极高和极低概率
    // 通过typical参数控制典型性阈值，平衡创造性和一致性
    subset = typical(subset);
  }

  if (sampler_type == "mixed") {
    // 混合采样器：按顺序应用多种采样策略的组合
    // 根据mixed_samplers配置依次执行多个采样器，实现复合过滤效果
    subset = mixed(subset);
  }
  // if greedy and temperate, just let the Selector handle it.
  return subset;
}

int Sampler::handleSelect(struct SubsetLogits subset) {
  if ((dynamic_cast<SampleParam*>(param_.get()))->sampler == "greedy") {
    return argmaxSelect(subset);
  } else if ((dynamic_cast<SampleParam*>(param_.get()))->sampler ==
             "temperature") {
    return reSoftmaxSelect(
        subset, (dynamic_cast<SampleParam*>(param_.get()))->temperature);
  }
  return 0;
}

int Sampler::sample(device::Tensor* logits) {
  // create subset logits
  struct SubsetLogits subset = createSubsetLogits(logits);
  // process subsetSampler
  SampleParam* sample_param = dynamic_cast<SampleParam*>(param_.get());
  subset = subsetSampler(sample_param->sampler, subset);
  // select token from the subset
  int res = handleSelect(subset);
  return res;
}

REGISTER_NODE("nndeploy::llm::Sampler", Sampler);

}  // namespace llm
}  // namespace nndeploy