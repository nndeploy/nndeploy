
#ifndef _NNDEPLOY_LLM_ABSTRACT_LLM_INFER_H_
#define _NNDEPLOY_LLM_ABSTRACT_LLM_INFER_H_

#include "nndeploy/base/any.h"
#include "nndeploy/base/common.h"
#include "nndeploy/base/glic_stl_include.h"
#include "nndeploy/base/log.h"
#include "nndeploy/base/macro.h"
#include "nndeploy/base/object.h"
#include "nndeploy/base/opencv_include.h"
#include "nndeploy/base/param.h"
#include "nndeploy/base/status.h"
#include "nndeploy/base/string.h"
#include "nndeploy/dag/composite_node.h"
#include "nndeploy/dag/edge.h"
#include "nndeploy/dag/graph.h"
#include "nndeploy/dag/loop.h"
#include "nndeploy/dag/node.h"
#include "nndeploy/device/buffer.h"
#include "nndeploy/device/device.h"
#include "nndeploy/device/memory_pool.h"
#include "nndeploy/device/tensor.h"
#include "nndeploy/infer/infer.h"
#include "nndeploy/tokenizer/tokenizer.h"

namespace nndeploy {
namespace llm {

class AbstractLlmInfer : dag::CompositeNode {
 public:
  AbstractLlmInfer(const std::string &name) : dag::CompositeNode(name) {
    key_ = "nndeploy::llm::LlmInfer";
    desc_ =
        "LLM abstract pipeline: input_ids -> "
        "inference -> [logits]";
    this->setInputTypeInfo<tokenizer::TokenizerIds>("input_tokens");
    this->setOutputTypeInfo<device::Tensor>("output_logits");
  }
  AbstractLlmInfer(const std::string &name, std::vector<dag::Edge *> inputs,
                   std::vector<dag::Edge *> outputs)
      : dag::CompositeNode(name, inputs, outputs) {
    key_ = "nndeploy::llm::LlmInfer";
    desc_ =
        "LLM abstract pipeline: input_ids -> "
        "inference -> [logits]";
    this->setInputTypeInfo<tokenizer::TokenizerIds>("input_tokens");
    this->setOutputTypeInfo<device::Tensor>("output_logits");
  }
  virtual ~AbstractLlmInfer() {}

  virtual base::Status init() {
    return base::kStatusCodeOk;
  }
  virtual base::Status deinit() {
    return base::kStatusCodeOk;
  }

  virtual base::Status run() = 0;

  virtual base::Status setPrefill(bool is_prefill) {
    is_prefill_ = is_prefill;
    return base::kStatusCodeOk;
  }
  virtual base::Status setConfigPath(
      const std::vector<std::string> &config_path) {
    config_path_ = config_path;
    return base::kStatusCodeOk;
  }
  virtual base::Status setModelKey(const std::string &model_key) {
    model_key_ = model_key;
    return base::kStatusCodeOk;
  }
  virtual base::Status setInferKey(const std::string &infer_key) {
    infer_key_ = infer_key;
    return base::kStatusCodeOk;
  }

  device::Tensor *genPastKeyValue(const std::vector<int32_t> &kv_init_shape,
                                  base::DeviceType device_type) {
    device::Device *device = device::getDevice(device_type);
    device::TensorDesc past_kv_desc;
    past_kv_desc.data_type_ = base::dataTypeOf<float>();
    past_kv_desc.data_format_ = base::DataFormat::kDataFormatS1D;
    past_kv_desc.shape_ = kv_init_shape;
    device::Tensor *past_kv;
    past_kv = new device::Tensor(device, past_kv_desc, "past_key_values");
    return past_kv;
  }

  device::Tensor *genPositionIds(int seq_len, int all_seq_len,
                                 base::DataType data_type,
                                 base::DataFormat data_format,
                                 base::DeviceType device_type) {
    device::Device *device = device::getDevice(device_type);
    device::TensorDesc position_ids_desc;
    position_ids_desc.data_type_ = data_type;
    position_ids_desc.data_format_ = data_format;
    position_ids_desc.shape_ = {1, seq_len};
    device::Tensor *position_ids;
    position_ids =
        new device::Tensor(device, position_ids_desc, "position_ids");

    // only host
    auto ptr = (int *)position_ids->getData();
    if (seq_len == 1) {
      ptr[0] = all_seq_len;
    } else {
      for (int i = 0; i < seq_len; i++) {
        ptr[i] = i + all_seq_len;
      }
    }

    return position_ids;
  }

  device::Tensor *genAttentionMask(int seq_len, int all_seq_len,
                                   base::DataType data_type,
                                   base::DataFormat data_format,
                                   base::DeviceType device_type) {
    int kv_seq_len = all_seq_len + seq_len;
    if (seq_len == 1) kv_seq_len = seq_len;

    /* create attetion_mask tensor */
    device::Device *device = device::getDevice(device_type);
    device::TensorDesc attention_mask_desc;
    attention_mask_desc.data_type_ = data_type;
    attention_mask_desc.data_format_ = data_format;
    attention_mask_desc.shape_ = {1, 1, seq_len, kv_seq_len};
    device::Tensor *attention_mask;
    attention_mask =
        new device::Tensor(device, attention_mask_desc, "attention_mask");

    // only host
    auto ptr = (float *)attention_mask->getData();
    for (int i = 0; i < seq_len; i++) {
      for (int j = 0; j < kv_seq_len; j++) {
        int row = i + all_seq_len;
        ptr[kv_seq_len * i + j] =
            (j > row) * std::numeric_limits<float>::lowest();
      }
    }

    return attention_mask;
  }

 protected:
  // prefill or decode
  bool is_prefill_ = true;
  // config_path
  std::vector<std::string> config_path_;
  // qwen or llama...
  std::string model_key_;
  // llm::DefaultLlmInfer or llm::MnnLlmInfer
  std::string infer_key_;

  // 输入
  std::string input_ids_name_ = "input_ids";
  std::string attention_mask_name_ = "attention_mask";
  std::string position_ids_name_ = "position_ids";
  std::string past_key_values_name_ = "past_key_values";
  // 输出
  std::string logits_name_ = "logits";
  std::string presents_name_ = "presents";
};

// 前向声明
template <typename T>
class TypeLlmInferCreator;

class NNDEPLOY_CC_API LlmInferCreator {
 public:
  virtual ~LlmInferCreator() = default;
  virtual AbstractLlmInfer *createLlmInfer(const std::string &name) = 0;
  virtual AbstractLlmInfer *createLlmInfer(
      const std::string &name, std::vector<dag::Edge *> inputs,
      std::vector<dag::Edge *> outputs) = 0;
};

template <typename T>
class TypeLlmInferCreator : public LlmInferCreator {
 public:
  virtual AbstractLlmInfer *createLlmInfer(const std::string &name) override {
    return new T(name);
  }
  virtual AbstractLlmInfer *createLlmInfer(
      const std::string &name, std::vector<dag::Edge *> inputs,
      std::vector<dag::Edge *> outputs) override {
    return new T(name, inputs, outputs);
  }
};

class NNDEPLOY_CC_API LlmInferFactory {
 public:
  static LlmInferFactory *getInstance() {
    static LlmInferFactory instance;
    return &instance;
  }

  void registerLlmInfer(const std::string &infer_key,
                        const std::string &model_key,
                        std::shared_ptr<LlmInferCreator> creator) {
    auto it = creators_.find(infer_key);
    if (it == creators_.end()) {
      creators_[infer_key] =
          std::map<std::string, std::shared_ptr<LlmInferCreator>>();
    }

    auto model_it = creators_[infer_key].find(model_key);
    if (model_it != creators_[infer_key].end()) {
      NNDEPLOY_LOGW("LlmInfer %s@%s already exists, will be overwritten!\n",
                    infer_key.c_str(), model_key.c_str());
    }
    creators_[infer_key][model_key] = creator;
    NNDEPLOY_LOGI("register LlmInfer success: %s@%s\n", infer_key.c_str(),
                  model_key.c_str());
  }

  std::shared_ptr<LlmInferCreator> getCreator(const std::string &infer_key,
                                              const std::string &model_key) {
    auto infer_it = creators_.find(infer_key);
    if (infer_it != creators_.end()) {
      auto model_it = infer_it->second.find(model_key);
      if (model_it != infer_it->second.end()) {
        return model_it->second;
      }
    }
    NNDEPLOY_LOGE("LlmInfer %s@%s not found!\n", infer_key.c_str(),
                  model_key.c_str());
    return nullptr;
  }

  std::set<std::string> getInferKeys() {
    std::set<std::string> keys;
    for (auto &it : creators_) {
      keys.insert(it.first);
    }
    return keys;
  }

  std::set<std::string> getModelKeys(const std::string &infer_key) {
    std::set<std::string> keys;
    auto it = creators_.find(infer_key);
    if (it != creators_.end()) {
      for (auto &model_it : it->second) {
        keys.insert(model_it.first);
      }
    }
    return keys;
  }

  std::set<std::pair<std::string, std::string>> getAllKeys() {
    std::set<std::pair<std::string, std::string>> keys;
    for (auto &infer_it : creators_) {
      for (auto &model_it : infer_it.second) {
        keys.insert(std::make_pair(infer_it.first, model_it.first));
      }
    }
    return keys;
  }

 private:
  LlmInferFactory() = default;
  ~LlmInferFactory() = default;
  std::map<std::string, std::map<std::string, std::shared_ptr<LlmInferCreator>>>
      creators_;
};

#define REGISTER_LLM_INFER(infer_key, model_key, node_class)                   \
  namespace {                                                                  \
  struct LlmInferRegister_##node_class {                                       \
    LlmInferRegister_##node_class() {                                          \
      LlmInferFactory::getInstance()->registerLlmInfer(                        \
          infer_key, model_key,                                                \
          std::make_shared<nndeploy::llm::TypeLlmInferCreator<node_class>>()); \
    }                                                                          \
  };                                                                           \
  static LlmInferRegister_##node_class g_llm_infer_register_##node_class;      \
  }

}  // namespace llm
}  // namespace nndeploy

#endif