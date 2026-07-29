#include "nndeploy/segment/segment_anything/sam3.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <memory>
#include <unordered_map>

#include "nndeploy/infer/infer.h"
#include "nndeploy/preprocess/cvt_resize_pad_norm_trans.h"

namespace nndeploy {
namespace segment {

#define CHECK_IF_NULL_RETURN(ptr, fmt, ...)    \
  if (ptr == nullptr) {                        \
    NNDEPLOY_LOGE(fmt, ##__VA_ARGS__);         \
    return base::kStatusCodeErrorInvalidValue; \
  }

#define CHECK_IF_ERROR_RETURN(ret, fmt, ...) \
  if (ret != base::kStatusCodeOk) {          \
    NNDEPLOY_LOGE(fmt, ##__VA_ARGS__);       \
    return ret;                              \
  }

// ===================== Sam3ConceptParam =====================

base::Status Sam3ConceptParam::serialize(
    rapidjson::Value& json, rapidjson::Document::AllocatorType& allocator) {
  json.SetObject();
  rapidjson::Value concepts_array(rapidjson::kArrayType);
  for (const auto& c : concepts_) {
    rapidjson::Value v;
    v.SetString(c.c_str(), allocator);
    concepts_array.PushBack(v, allocator);
  }
  json.AddMember("concepts", concepts_array, allocator);
  json.AddMember("num_concepts_", num_concepts_, allocator);
  json.AddMember("concept_dim_", concept_dim_, allocator);
  json.AddMember("ori_width_", ori_width_, allocator);
  json.AddMember("ori_height_", ori_height_, allocator);
  return base::kStatusCodeOk;
}

base::Status Sam3ConceptParam::deserialize(rapidjson::Value& json) {
  if (json.HasMember("concepts") && json["concepts"].IsArray()) {
    concepts_.clear();
    const auto& arr = json["concepts"];
    for (rapidjson::SizeType i = 0; i < arr.Size(); i++) {
      if (arr[i].IsString()) {
        concepts_.push_back(arr[i].GetString());
      }
    }
  }
  if (json.HasMember("num_concepts_") && json["num_concepts_"].IsInt()) {
    num_concepts_ = json["num_concepts_"].GetInt();
  }
  if (json.HasMember("concept_dim_") && json["concept_dim_"].IsInt()) {
    concept_dim_ = json["concept_dim_"].GetInt();
  }
  if (json.HasMember("ori_width_") && json["ori_width_"].IsInt()) {
    ori_width_ = json["ori_width_"].GetInt();
  }
  if (json.HasMember("ori_height_") && json["ori_height_"].IsInt()) {
    ori_height_ = json["ori_height_"].GetInt();
  }
  return base::kStatusCodeOk;
}

// ===================== Sam3LanguageParam =====================

base::Status Sam3LanguageParam::serialize(
    rapidjson::Value& json, rapidjson::Document::AllocatorType& allocator) {
  json.SetObject();
  json.AddMember("max_token_length_", max_token_length_, allocator);
  json.AddMember("hidden_dim_", hidden_dim_, allocator);
  json.AddMember("num_concepts_", num_concepts_, allocator);
  return base::kStatusCodeOk;
}

base::Status Sam3LanguageParam::deserialize(rapidjson::Value& json) {
  if (json.HasMember("max_token_length_") &&
      json["max_token_length_"].IsInt()) {
    max_token_length_ = json["max_token_length_"].GetInt();
  }
  if (json.HasMember("hidden_dim_") && json["hidden_dim_"].IsInt()) {
    hidden_dim_ = json["hidden_dim_"].GetInt();
  }
  if (json.HasMember("num_concepts_") && json["num_concepts_"].IsInt()) {
    num_concepts_ = json["num_concepts_"].GetInt();
  }
  return base::kStatusCodeOk;
}

// ===================== Sam3ExemplarParam =====================

base::Status Sam3ExemplarParam::serialize(
    rapidjson::Value& json, rapidjson::Document::AllocatorType& allocator) {
  json.SetObject();
  json.AddMember("num_exemplars_", num_exemplars_, allocator);
  return base::kStatusCodeOk;
}

base::Status Sam3ExemplarParam::deserialize(rapidjson::Value& json) {
  if (json.HasMember("num_exemplars_") && json["num_exemplars_"].IsInt()) {
    num_exemplars_ = json["num_exemplars_"].GetInt();
  }
  return base::kStatusCodeOk;
}

// ===================== Sam3PostParam =====================

base::Status Sam3PostParam::serialize(
    rapidjson::Value& json, rapidjson::Document::AllocatorType& allocator) {
  json.SetObject();
  json.AddMember("score_threshold_", score_threshold_, allocator);
  json.AddMember("presence_threshold_", presence_threshold_, allocator);
  json.AddMember("concept_threshold_", concept_threshold_, allocator);
  json.AddMember("model_h_", model_h_, allocator);
  json.AddMember("model_w_", model_w_, allocator);
  return base::kStatusCodeOk;
}

base::Status Sam3PostParam::deserialize(rapidjson::Value& json) {
  if (json.HasMember("score_threshold_") &&
      json["score_threshold_"].IsFloat()) {
    score_threshold_ = json["score_threshold_"].GetFloat();
  }
  if (json.HasMember("presence_threshold_") &&
      json["presence_threshold_"].IsFloat()) {
    presence_threshold_ = json["presence_threshold_"].GetFloat();
  }
  if (json.HasMember("concept_threshold_") &&
      json["concept_threshold_"].IsFloat()) {
    concept_threshold_ = json["concept_threshold_"].GetFloat();
  }
  if (json.HasMember("model_h_") && json["model_h_"].IsInt()) {
    model_h_ = json["model_h_"].GetInt();
  }
  if (json.HasMember("model_w_") && json["model_w_"].IsInt()) {
    model_w_ = json["model_w_"].GetInt();
  }
  return base::kStatusCodeOk;
}

// ===================== New Architecture Param Classes =====================

base::Status Sam3PerceptionEncoderParam::serialize(
    rapidjson::Value& json, rapidjson::Document::AllocatorType& allocator) {
  json.SetObject();
  json.AddMember("embed_dim_", embed_dim_, allocator);
  json.AddMember("image_size_", image_size_, allocator);
  json.AddMember("use_high_res_", use_high_res_, allocator);
  return base::kStatusCodeOk;
}

base::Status Sam3PerceptionEncoderParam::deserialize(rapidjson::Value& json) {
  if (json.HasMember("embed_dim_") && json["embed_dim_"].IsInt()) {
    embed_dim_ = json["embed_dim_"].GetInt();
  }
  if (json.HasMember("image_size_") && json["image_size_"].IsInt()) {
    image_size_ = json["image_size_"].GetInt();
  }
  if (json.HasMember("use_high_res_") && json["use_high_res_"].IsBool()) {
    use_high_res_ = json["use_high_res_"].GetBool();
  }
  return base::kStatusCodeOk;
}

base::Status Sam3DetectorDecoderParam::serialize(
    rapidjson::Value& json, rapidjson::Document::AllocatorType& allocator) {
  json.SetObject();
  json.AddMember("num_queries_", num_queries_, allocator);
  json.AddMember("query_dim_", query_dim_, allocator);
  json.AddMember("has_presence_token_", has_presence_token_, allocator);
  json.AddMember("num_decoder_layers_", num_decoder_layers_, allocator);
  json.AddMember("box_refine_", box_refine_, allocator);
  return base::kStatusCodeOk;
}

base::Status Sam3DetectorDecoderParam::deserialize(rapidjson::Value& json) {
  if (json.HasMember("num_queries_") && json["num_queries_"].IsInt()) {
    num_queries_ = json["num_queries_"].GetInt();
  }
  if (json.HasMember("query_dim_") && json["query_dim_"].IsInt()) {
    query_dim_ = json["query_dim_"].GetInt();
  }
  if (json.HasMember("has_presence_token_") &&
      json["has_presence_token_"].IsBool()) {
    has_presence_token_ = json["has_presence_token_"].GetBool();
  }
  if (json.HasMember("num_decoder_layers_") &&
      json["num_decoder_layers_"].IsInt()) {
    num_decoder_layers_ = json["num_decoder_layers_"].GetInt();
  }
  if (json.HasMember("box_refine_") && json["box_refine_"].IsBool()) {
    box_refine_ = json["box_refine_"].GetBool();
  }
  return base::kStatusCodeOk;
}

base::Status Sam3PresenceHeadParam::serialize(
    rapidjson::Value& json, rapidjson::Document::AllocatorType& allocator) {
  json.SetObject();
  json.AddMember("presence_threshold_", presence_threshold_, allocator);
  json.AddMember("enable_nms_", enable_nms_, allocator);
  json.AddMember("nms_threshold_", nms_threshold_, allocator);
  return base::kStatusCodeOk;
}

base::Status Sam3PresenceHeadParam::deserialize(rapidjson::Value& json) {
  if (json.HasMember("presence_threshold_") &&
      json["presence_threshold_"].IsFloat()) {
    presence_threshold_ = json["presence_threshold_"].GetFloat();
  }
  if (json.HasMember("enable_nms_") && json["enable_nms_"].IsBool()) {
    enable_nms_ = json["enable_nms_"].GetBool();
  }
  if (json.HasMember("nms_threshold_") && json["nms_threshold_"].IsFloat()) {
    nms_threshold_ = json["nms_threshold_"].GetFloat();
  }
  return base::kStatusCodeOk;
}

base::Status Sam3ConceptMatcherParam::serialize(
    rapidjson::Value& json, rapidjson::Document::AllocatorType& allocator) {
  json.SetObject();
  json.AddMember("similarity_threshold_", similarity_threshold_, allocator);
  json.AddMember("scoring_type_",
                 rapidjson::Value(scoring_type_.c_str(), allocator), allocator);
  json.AddMember("per_concept_nms_", per_concept_nms_, allocator);
  return base::kStatusCodeOk;
}

base::Status Sam3ConceptMatcherParam::deserialize(rapidjson::Value& json) {
  if (json.HasMember("similarity_threshold_") &&
      json["similarity_threshold_"].IsFloat()) {
    similarity_threshold_ = json["similarity_threshold_"].GetFloat();
  }
  if (json.HasMember("scoring_type_") && json["scoring_type_"].IsString()) {
    scoring_type_ = json["scoring_type_"].GetString();
  }
  if (json.HasMember("per_concept_nms_") && json["per_concept_nms_"].IsBool()) {
    per_concept_nms_ = json["per_concept_nms_"].GetBool();
  }
  return base::kStatusCodeOk;
}

base::Status Sam3MemoryEncoderParam::serialize(
    rapidjson::Value& json, rapidjson::Document::AllocatorType& allocator) {
  json.SetObject();
  json.AddMember("memory_dim_", memory_dim_, allocator);
  json.AddMember("max_memory_frames_", max_memory_frames_, allocator);
  return base::kStatusCodeOk;
}

base::Status Sam3MemoryEncoderParam::deserialize(rapidjson::Value& json) {
  if (json.HasMember("memory_dim_") && json["memory_dim_"].IsInt()) {
    memory_dim_ = json["memory_dim_"].GetInt();
  }
  if (json.HasMember("max_memory_frames_") &&
      json["max_memory_frames_"].IsInt()) {
    max_memory_frames_ = json["max_memory_frames_"].GetInt();
  }
  return base::kStatusCodeOk;
}

base::Status Sam3TrackerMaskDecoderParam::serialize(
    rapidjson::Value& json, rapidjson::Document::AllocatorType& allocator) {
  json.SetObject();
  json.AddMember("num_mask_embeddings_", num_mask_embeddings_, allocator);
  json.AddMember("embedding_dim_", embedding_dim_, allocator);
  return base::kStatusCodeOk;
}

base::Status Sam3TrackerMaskDecoderParam::deserialize(rapidjson::Value& json) {
  if (json.HasMember("num_mask_embeddings_") &&
      json["num_mask_embeddings_"].IsInt()) {
    num_mask_embeddings_ = json["num_mask_embeddings_"].GetInt();
  }
  if (json.HasMember("embedding_dim_") && json["embedding_dim_"].IsInt()) {
    embedding_dim_ = json["embedding_dim_"].GetInt();
  }
  return base::kStatusCodeOk;
}

base::Status Sam3MemoryManagerParam::serialize(
    rapidjson::Value& json, rapidjson::Document::AllocatorType& allocator) {
  json.SetObject();
  json.AddMember("max_memory_frames_", max_memory_frames_, allocator);
  json.AddMember("frame_stride_", frame_stride_, allocator);
  json.AddMember("enable_memory_temperature_", enable_memory_temperature_,
                 allocator);
  return base::kStatusCodeOk;
}

base::Status Sam3MemoryManagerParam::deserialize(rapidjson::Value& json) {
  if (json.HasMember("max_memory_frames_") &&
      json["max_memory_frames_"].IsInt()) {
    max_memory_frames_ = json["max_memory_frames_"].GetInt();
  }
  if (json.HasMember("frame_stride_") && json["frame_stride_"].IsInt()) {
    frame_stride_ = json["frame_stride_"].GetInt();
  }
  if (json.HasMember("enable_memory_temperature_") &&
      json["enable_memory_temperature_"].IsBool()) {
    enable_memory_temperature_ = json["enable_memory_temperature_"].GetBool();
  }
  return base::kStatusCodeOk;
}

// ===================== Simplified ONNX Node Param Classes
// =====================

base::Status Sam3SimpleImageEncoderParam::serialize(
    rapidjson::Value& json, rapidjson::Document::AllocatorType& allocator) {
  json.SetObject();
  json.AddMember("image_height_", image_height_, allocator);
  json.AddMember("image_width_", image_width_, allocator);
  return base::kStatusCodeOk;
}

base::Status Sam3SimpleImageEncoderParam::deserialize(rapidjson::Value& json) {
  if (json.HasMember("image_height_") && json["image_height_"].IsInt()) {
    image_height_ = json["image_height_"].GetInt();
  }
  if (json.HasMember("image_width_") && json["image_width_"].IsInt()) {
    image_width_ = json["image_width_"].GetInt();
  }
  return base::kStatusCodeOk;
}

base::Status Sam3SimpleLanguageEncoderParam::serialize(
    rapidjson::Value& json, rapidjson::Document::AllocatorType& allocator) {
  json.SetObject();
  json.AddMember("max_token_length_", max_token_length_, allocator);
  return base::kStatusCodeOk;
}

base::Status Sam3SimpleLanguageEncoderParam::deserialize(
    rapidjson::Value& json) {
  if (json.HasMember("max_token_length_") &&
      json["max_token_length_"].IsInt()) {
    max_token_length_ = json["max_token_length_"].GetInt();
  }
  return base::kStatusCodeOk;
}

base::Status Sam3SimpleDecoderParam::serialize(
    rapidjson::Value& json, rapidjson::Document::AllocatorType& allocator) {
  json.SetObject();
  json.AddMember("score_threshold_", score_threshold_, allocator);
  return base::kStatusCodeOk;
}

base::Status Sam3SimpleDecoderParam::deserialize(rapidjson::Value& json) {
  if (json.HasMember("score_threshold_") &&
      json["score_threshold_"].IsFloat()) {
    score_threshold_ = json["score_threshold_"].GetFloat();
  }
  return base::kStatusCodeOk;
}

// ===================== Sam3SimpleImageEncoder =====================

base::Status Sam3SimpleImageEncoder::defaultParam() {
  Sam3SimpleImageEncoderParam* param =
      dynamic_cast<Sam3SimpleImageEncoderParam*>(param_.get());
  param->image_height_ = 1008;
  param->image_width_ = 1008;
  return base::kStatusCodeOk;
}

base::Status Sam3SimpleImageEncoder::setInferParam(
    base::InferenceType inference_type, base::DeviceType device_type,
    base::ModelType model_type, bool is_path, std::string& model_path) {
  image_encoder_param_.inference_type_ = inference_type;
  image_encoder_param_.device_type_ = device_type;
  image_encoder_param_.model_type_ = model_type;
  image_encoder_param_.is_path_ = is_path;
  image_encoder_param_.model_value_ = {model_path};

  if (image_encoder_infer_ == nullptr) {
    image_encoder_infer_ = new infer::Infer("sam3_image_encoder");
  }
  base::Status status = image_encoder_infer_->setInferenceType(inference_type);
  CHECK_IF_ERROR_RETURN(status, "Failed to set image encoder inference type");
  status = image_encoder_infer_->setParam(&image_encoder_param_);
  CHECK_IF_ERROR_RETURN(status, "Failed to set image encoder param");
  return status;
}

base::Status Sam3SimpleImageEncoder::run() {
  if (image_encoder_infer_ == nullptr) {
    NNDEPLOY_LOGE("Sam3SimpleImageEncoder: no infer node configured.");
    return base::kStatusCodeErrorInvalidValue;
  }

  device::Tensor* input_tensor = inputs_[0]->getTensor(this);
  CHECK_IF_NULL_RETURN(input_tensor, "Image encoder input tensor is null");

  // Directly use the inference session instead of going through Infer's edge
  // system — the Infer node was created standalone (no DAG edges) so
  // getInput/getOutput on it would crash on nullptr.
  auto inference = image_encoder_infer_->getInference();
  CHECK_IF_NULL_RETURN(inference, "Image encoder inference session is null");

  // Get input tensor name from the model
  auto input_names = inference->getAllInputTensorName();
  if (input_names.empty()) {
    NNDEPLOY_LOGE("Sam3SimpleImageEncoder: no input names in inference session");
    return base::kStatusCodeErrorInvalidValue;
  }
  const std::string& input_name = input_names[0];

  // The preprocess produces NCHW [1, 3, 1008, 1008] uint8, but the ONNX model
  // expects CHW [3, 1008, 1008] uint8 (no batch dim). Squeeze the batch
  // dimension by creating a view that wraps the same buffer with the CHW desc.
  // The view tensor must stay alive until after inference->run() because the
  // inference session stores a raw pointer to it in external_input_tensors_.
  device::TensorDesc input_desc = input_tensor->getDesc();
  device::Tensor* model_input = input_tensor;
  bool needs_squeeze =
      (input_desc.shape_.size() == 4 && input_desc.shape_[0] == 1);
  std::unique_ptr<device::Tensor> squeeze_holder;
  if (needs_squeeze) {
    device::TensorDesc chw_desc = input_desc;
    chw_desc.shape_.erase(chw_desc.shape_.begin());  // drop batch dim
    squeeze_holder.reset(
        new device::Tensor(chw_desc, input_tensor->getBuffer(), input_name));
    model_input = squeeze_holder.get();
  } else {
    model_input->setName(input_name);
  }

  base::Status status = inference->setInputTensor(input_name, model_input);
  CHECK_IF_ERROR_RETURN(status, "Image encoder setInputTensor failed");

  status = inference->run();
  CHECK_IF_ERROR_RETURN(status, "Image encoder inference failed");

  base::DeviceType device_type = inference->getDeviceType();
  auto output_names = inference->getAllOutputTensorName();

  int output_count = static_cast<int>(output_names.size());
  if (output_count < 6) {
    NNDEPLOY_LOGE("Sam3SimpleImageEncoder: expected 6 outputs, got %d",
                  output_count);
    return base::kStatusCodeErrorInvalidValue;
  }

  device::Device* cur_device = device::getDefaultHostDevice();
  // NOTE: getAllOutputTensorName iterates over an internal std::map (alphabetical
  // key order), so the returned vector order differs from the ONNX declaration.
  // Match by name rather than by positional index.
  for (int i = 0; i < static_cast<int>(output_names.size()); ++i) {
    // Map ONNX output name → the target edge index (outputs_[N])
    int out_idx = -1;
    if (output_names[i] == "vision_pos_enc_0") {
      out_idx = 0;
    } else if (output_names[i] == "vision_pos_enc_1") {
      out_idx = 1;
    } else if (output_names[i] == "vision_pos_enc_2") {
      out_idx = 2;
    } else if (output_names[i] == "backbone_fpn_0") {
      out_idx = 3;
    } else if (output_names[i] == "backbone_fpn_1") {
      out_idx = 4;
    } else if (output_names[i] == "backbone_fpn_2") {
      out_idx = 5;
    } else {
      continue;  // Skip unexpected outputs
    }

    device::Tensor* out_tensor = inference->getOutputTensorAfterRun(
        output_names[i], device_type, false);
    CHECK_IF_NULL_RETURN(out_tensor, "Image encoder output %s is null",
                         output_names[i].c_str());

    device::TensorDesc out_desc = out_tensor->getDesc();
    // The ONNX model expects CHW input (no batch dim), so it treats the
    // first dim (3 channels) as batch, producing batch-3 outputs.
    // We must restore batch=1 for the decoder.
    size_t elem_size = out_desc.data_type_.size();
    size_t one_batch_bytes = elem_size;
    for (size_t d = 1; d < out_desc.shape_.size(); ++d) {
      one_batch_bytes *= out_desc.shape_[d];
    }
    out_desc.shape_[0] = 1;
    device::Tensor* out = outputs_[out_idx]->create(cur_device, out_desc);
    CHECK_IF_NULL_RETURN(out, "Failed to create image encoder output %d",
                         out_idx);
    if (out->getData() != nullptr && out_tensor->getData() != nullptr) {
      memcpy(out->getData(), out_tensor->getData(), one_batch_bytes);
    }
  }
  return base::kStatusCodeOk;
}

// ===================== Sam3SimpleLanguageEncoder =====================

base::Status Sam3SimpleLanguageEncoder::defaultParam() {
  Sam3SimpleLanguageEncoderParam* param =
      dynamic_cast<Sam3SimpleLanguageEncoderParam*>(param_.get());
  param->max_token_length_ = 32;
  return base::kStatusCodeOk;
}

base::Status Sam3SimpleLanguageEncoder::setInferParam(
    base::InferenceType inference_type, base::DeviceType device_type,
    base::ModelType model_type, bool is_path, std::string& model_path) {
  language_encoder_param_.inference_type_ = inference_type;
  language_encoder_param_.device_type_ = device_type;
  language_encoder_param_.model_type_ = model_type;
  language_encoder_param_.is_path_ = is_path;
  language_encoder_param_.model_value_ = {model_path};

  if (language_encoder_infer_ == nullptr) {
    language_encoder_infer_ = new infer::Infer("sam3_language_encoder");
  }
  base::Status status =
      language_encoder_infer_->setInferenceType(inference_type);
  CHECK_IF_ERROR_RETURN(status,
                        "Failed to set language encoder inference type");
  status = language_encoder_infer_->setParam(&language_encoder_param_);
  CHECK_IF_ERROR_RETURN(status, "Failed to set language encoder param");
  return status;
}

base::Status Sam3SimpleLanguageEncoder::run() {
  if (language_encoder_infer_ == nullptr) {
    NNDEPLOY_LOGE("Sam3SimpleLanguageEncoder: no infer node configured.");
    return base::kStatusCodeErrorInvalidValue;
  }

  device::Tensor* tokens_tensor = inputs_[0]->getTensor(this);
  CHECK_IF_NULL_RETURN(tokens_tensor, "Language encoder tokens tensor is null");

  // Directly use the inference session (same reason as ImageEncoder — the
  // Infer node was created standalone and has no DAG edges).
  auto inference = language_encoder_infer_->getInference();
  CHECK_IF_NULL_RETURN(inference, "Language encoder inference session is null");

  auto input_names = inference->getAllInputTensorName();
  if (input_names.empty()) {
    NNDEPLOY_LOGE("Sam3SimpleLanguageEncoder: no input names");
    return base::kStatusCodeErrorInvalidValue;
  }
  tokens_tensor->setName(input_names[0]);
  base::Status status = inference->setInputTensor(input_names[0], tokens_tensor);
  CHECK_IF_ERROR_RETURN(status, "Language encoder setInputTensor failed");

  status = inference->run();
  CHECK_IF_ERROR_RETURN(status, "Language encoder inference failed");

  base::DeviceType device_type = inference->getDeviceType();
  auto output_names = inference->getAllOutputTensorName();

  int output_count = static_cast<int>(output_names.size());
  if (output_count < 3) {
    NNDEPLOY_LOGE("Sam3SimpleLanguageEncoder: expected 3 outputs, got %d",
                  output_count);
    return base::kStatusCodeErrorInvalidValue;
  }

  device::Device* cur_device = device::getDefaultHostDevice();
  // IMPORTANT: getAllOutputTensorName iterates over an internal std::map which
  // sorts keys alphabetically, producing [text_attention_mask, text_embeds,
  // text_memory] instead of the ONNX graph declaration order.  We must match
  // by name rather than by positional index.
  for (int i = 0; i < static_cast<int>(output_names.size()); ++i) {
    device::Tensor* out_tensor = inference->getOutputTensorAfterRun(
        output_names[i], device_type, false);
    CHECK_IF_NULL_RETURN(out_tensor, "Language encoder output %s is null",
                         output_names[i].c_str());

    // Determine output edge index by tensor name
    int out_idx = -1;
    if (output_names[i].find("text_attention_mask") != std::string::npos) {
      out_idx = 0;  // → outputs_[0] = lang_mask_edge
    } else if (output_names[i].find("text_memory") != std::string::npos) {
      out_idx = 1;  // → outputs_[1] = lang_features_edge
    } else {
      // Skip text_embeds — not needed by the decoder
      continue;
    }

    device::TensorDesc out_desc = out_tensor->getDesc();

    // Fix ONNX bool → nndeploy uint8 conversion for ONNX bool outputs.
    // ONNX TensorDataType BOOL is mapped to kDataTypeCodeUint + bits=8 by
    // convertToDataType, but nndeploy represents bool as kDataTypeCodeOpaqueHandle
    // (via dataTypeOf<bool>()).  The decoder expects OpaqueHandle for bool-like
    // inputs so convertFromDataType maps them to ONNX_TENSOR_ELEMENT_DATA_TYPE_BOOL.
    if (out_idx == 0 && out_desc.data_type_.code_ == base::kDataTypeCodeUint &&
        out_desc.data_type_.bits_ == 8 && out_desc.data_type_.lanes_ == 1) {
      out_desc.data_type_ = base::DataType(base::kDataTypeCodeOpaqueHandle, 8, 1);
    }

    // text_memory: ONNX declared [32,1,256] but the model produces [1,1,256]
    // at runtime (only position 0).  The decoder was initialized with static
    // shape [32,1,256] and rejects [1,1,256] via ORT's dimension check.
    // Tile the single position 32 times to match.
    bool need_tile = (out_idx == 1 && out_desc.shape_.size() == 3 &&
                      out_desc.shape_[0] == 1);
    if (need_tile) {
      out_desc.shape_[0] = 32;
    }

    device::Tensor* out = outputs_[out_idx]->create(cur_device, out_desc);
    CHECK_IF_NULL_RETURN(out, "Failed to create language encoder output %d",
                         out_idx);
    if (out->getData() != nullptr && out_tensor->getData() != nullptr) {
      if (need_tile) {
        // Copy the single position's data 32 times
        size_t pos_bytes = out_tensor->getSize();  // size of [1, 1, 256]
        for (int j = 0; j < 32; ++j) {
          memcpy(static_cast<char*>(out->getData()) + j * pos_bytes,
                 out_tensor->getData(), pos_bytes);
        }
      } else {
        memcpy(out->getData(), out_tensor->getData(), out_tensor->getSize());
      }
    }
  }
  return base::kStatusCodeOk;
}

// ===================== Sam3SimpleDecoder =====================

base::Status Sam3SimpleDecoder::defaultParam() {
  Sam3SimpleDecoderParam* param =
      dynamic_cast<Sam3SimpleDecoderParam*>(param_.get());
  param->score_threshold_ = 0.5f;
  return base::kStatusCodeOk;
}

base::Status Sam3SimpleDecoder::setInferParam(
    base::InferenceType inference_type, base::DeviceType device_type,
    base::ModelType model_type, bool is_path, std::string& model_path) {
  decoder_param_.inference_type_ = inference_type;
  decoder_param_.device_type_ = device_type;
  decoder_param_.model_type_ = model_type;
  decoder_param_.is_path_ = is_path;
  decoder_param_.model_value_ = {model_path};

  if (decoder_infer_ == nullptr) {
    decoder_infer_ = new infer::Infer("sam3_decoder");
  }
  base::Status status = decoder_infer_->setInferenceType(inference_type);
  CHECK_IF_ERROR_RETURN(status, "Failed to set decoder inference type");
  status = decoder_infer_->setParam(&decoder_param_);
  CHECK_IF_ERROR_RETURN(status, "Failed to set decoder param");
  return status;
}

base::Status Sam3SimpleDecoder::run() {
  if (decoder_infer_ == nullptr) {
    NNDEPLOY_LOGE("Sam3SimpleDecoder: no infer node configured.");
    return base::kStatusCodeErrorInvalidValue;
  }

  // Directly use the inference session (same reason as ImageEncoder — the
  // Infer node was created standalone and has no DAG edges).
  auto inference = decoder_infer_->getInference();
  CHECK_IF_NULL_RETURN(inference, "Decoder inference session is null");

  auto input_names = inference->getAllInputTensorName();
  if (input_names.empty()) {
    NNDEPLOY_LOGE("Sam3SimpleDecoder: no input names in inference session");
    return base::kStatusCodeErrorInvalidValue;
  }

  // Build a name→tensor map from the 11 input edges instead of relying on
  // positional matching.  The external edges are in our own assembly order
  // which differs from the ONNX model's native order.
  std::unordered_map<std::string, device::Tensor*> tensor_by_name;
  auto edge_names = {std::make_pair(std::string("backbone_fpn_0"),  3),
                     std::make_pair(std::string("backbone_fpn_1"),  4),
                     std::make_pair(std::string("backbone_fpn_2"),  5),
                     std::make_pair(std::string("box_coords"),      8),
                     std::make_pair(std::string("box_labels"),      9),
                     std::make_pair(std::string("box_masks"),      10),
                     std::make_pair(std::string("language_features"), 7),
                     std::make_pair(std::string("language_mask"),   6),
                     std::make_pair(std::string("original_height"), 0),
                     std::make_pair(std::string("original_width"),  1),
                     std::make_pair(std::string("vision_pos_enc_2"), 2)};
  for (const auto& kv : edge_names) {
    device::Tensor* t = inputs_[kv.second]->getTensor(this);
    if (t != nullptr) {
      tensor_by_name[kv.first] = t;
    } else {
      NNDEPLOY_LOGE("Sam3SimpleDecoder: edge \"%s\" (idx %d) has no tensor\n",
                    kv.first.c_str(), kv.second);
    }
  }

  base::Status status = base::kStatusCodeOk;
  for (size_t i = 0; i < input_names.size(); ++i) {
    auto it = tensor_by_name.find(input_names[i]);
    if (it == tensor_by_name.end()) {
      NNDEPLOY_LOGE("Sam3SimpleDecoder: no tensor for ONNX input \"%s\"\n",
                    input_names[i].c_str());
      return base::kStatusCodeErrorInvalidValue;
    }
    device::Tensor* input = it->second;
    input->setName(input_names[i]);
    status = inference->setInputTensor(input_names[i], input);
    CHECK_IF_ERROR_RETURN(status, "Decoder setInputTensor %s failed",
                          input_names[i].c_str());
  }

  status = inference->run();
  CHECK_IF_ERROR_RETURN(status, "Decoder inference failed");

  base::DeviceType device_type = inference->getDeviceType();
  auto output_names = inference->getAllOutputTensorName();

  int output_count = static_cast<int>(output_names.size());
  if (output_count < 3) {
    NNDEPLOY_LOGE("Sam3SimpleDecoder: expected 3 outputs, got %d",
                  output_count);
    return base::kStatusCodeErrorInvalidValue;
  }

  device::Device* cur_device = device::getDefaultHostDevice();
  // NOTE: getAllOutputTensorName iterates over an internal std::map (alphabetical
  // key order), so the returned vector order differs from the ONNX declaration.
  // Match by name rather than by positional index.
  for (int i = 0; i < static_cast<int>(output_names.size()); ++i) {
    // Map ONNX output name → target edge index (outputs_[N])
    int out_idx = -1;
    if (output_names[i] == "boxes") {
      out_idx = 0;
    } else if (output_names[i] == "scores") {
      out_idx = 1;
    } else if (output_names[i] == "masks") {
      out_idx = 2;
    } else {
      continue;  // Skip unexpected outputs
    }

    device::Tensor* out_tensor = inference->getOutputTensorAfterRun(
        output_names[i], device_type, false);
    CHECK_IF_NULL_RETURN(out_tensor, "Decoder output %s is null",
                         output_names[i].c_str());

    device::TensorDesc out_desc = out_tensor->getDesc();
    device::Tensor* out = outputs_[out_idx]->create(cur_device, out_desc);
    CHECK_IF_NULL_RETURN(out, "Failed to create decoder output %d", out_idx);
    if (out->getData() != nullptr && out_tensor->getData() != nullptr) {
      memcpy(out->getData(), out_tensor->getData(), out_tensor->getSize());
    }
  }
  return base::kStatusCodeOk;
}

// ===================== Sam3SimplePostprocess =====================

base::Status Sam3SimplePostprocess::defaultParam() {
  Sam3PostParam* param = dynamic_cast<Sam3PostParam*>(param_.get());
  param->score_threshold_ = 0.5f;
  param->presence_threshold_ = 0.0f;
  param->concept_threshold_ = 0.0f;
  param->model_h_ = 0;
  param->model_w_ = 0;
  return base::kStatusCodeOk;
}

base::Status Sam3SimplePostprocess::run() {
  Sam3PostParam* param = dynamic_cast<Sam3PostParam*>(param_.get());
  float score_threshold = param->score_threshold_;

  device::Tensor* boxes_tensor = inputs_[0]->getTensor(this);
  device::Tensor* scores_tensor = inputs_[1]->getTensor(this);
  device::Tensor* masks_tensor = inputs_[2]->getTensor(this);

  CHECK_IF_NULL_RETURN(boxes_tensor, "Postprocess: boxes tensor is null");
  CHECK_IF_NULL_RETURN(scores_tensor, "Postprocess: scores tensor is null");
  CHECK_IF_NULL_RETURN(masks_tensor, "Postprocess: masks tensor is null");

  device::TensorDesc scores_desc = scores_tensor->getDesc();
  int num_instances = scores_desc.shape_[0];
  int num_masks = (masks_tensor->getDesc().shape_.size() > 0)
                      ? masks_tensor->getDesc().shape_[0]
                      : 0;

  device::TensorDesc masks_desc = masks_tensor->getDesc();
  int mask_h = (masks_desc.shape_.size() > 2) ? masks_desc.shape_[2] : 0;
  int mask_w = (masks_desc.shape_.size() > 3) ? masks_desc.shape_[3] : 0;

  float* scores_data = static_cast<float*>(scores_tensor->getData());
  float* boxes_data = static_cast<float*>(boxes_tensor->getData());

  // Clamp instance count to available masks to avoid out-of-bounds reads
  if (num_masks < num_instances) {
    NNDEPLOY_LOGW("Postprocess: num_masks(%d) < num_instances(%d), clamping\n",
                  num_masks, num_instances);
    num_instances = num_masks;
  }

  if (num_instances == 0 || mask_h == 0 || mask_w == 0) {
    NNDEPLOY_LOGI("Postprocess: no detections (num=%d, mask=%dx%d), "
                  "outputting blank image\n",
                  num_instances, mask_h, mask_w);
    device::Device *cur_device = device::getDefaultHostDevice();
    cv::Mat *result = new cv::Mat(mask_h > 0 ? mask_h : 100,
                                  mask_w > 0 ? mask_w : 100,
                                  CV_8UC3, cv::Scalar(0, 0, 0));
    outputs_[0]->set(result, false);
    return base::kStatusCodeOk;
  }

  base::DataType masks_dtype = masks_tensor->getDesc().data_type_;
  bool masks_are_bool = (masks_dtype.code_ == base::kDataTypeCodeUint &&
                         masks_dtype.bits_ == 8);
  bool masks_are_fp32 = (masks_dtype.code_ == base::kDataTypeCodeFp &&
                         masks_dtype.bits_ == 32);

  // Create output visualization
  cv::Mat* result = new cv::Mat(mask_h, mask_w, CV_8UC3, cv::Scalar(0, 0, 0));
  std::vector<cv::Scalar> colors = {
      cv::Scalar(255, 0, 0),   cv::Scalar(0, 255, 0),   cv::Scalar(0, 0, 255),
      cv::Scalar(255, 255, 0), cv::Scalar(255, 0, 255), cv::Scalar(0, 255, 255),
      cv::Scalar(128, 0, 0),   cv::Scalar(0, 128, 0),   cv::Scalar(0, 0, 128),
      cv::Scalar(128, 128, 0),
  };

  int kept = 0;
  for (int i = 0; i < num_instances; ++i) {
    if (scores_data[i] < score_threshold) {
      continue;
    }

    cv::Scalar color = colors[kept % colors.size()];

    for (int y = 0; y < mask_h; ++y) {
      for (int x = 0; x < mask_w; ++x) {
        bool mask_value = false;
        if (masks_are_bool) {
          const bool* bool_ptr = static_cast<const bool*>(
              masks_tensor->getData());
          mask_value = bool_ptr[i * mask_h * mask_w + y * mask_w + x];
        } else if (masks_are_fp32) {
          const float* fp32_ptr = static_cast<const float*>(
              masks_tensor->getData());
          mask_value = (fp32_ptr[i * mask_h * mask_w + y * mask_w + x] > 0.5f);
        }
        if (mask_value) {
          result->at<cv::Vec3b>(y, x) = cv::Vec3b(
              static_cast<uint8_t>(color[0]), static_cast<uint8_t>(color[1]),
              static_cast<uint8_t>(color[2]));
        }
      }
    }

    // Draw bounding box
    if (boxes_data != nullptr) {
      float cx = boxes_data[i * 4 + 0];
      float cy = boxes_data[i * 4 + 1];
      float w = boxes_data[i * 4 + 2];
      float h = boxes_data[i * 4 + 3];
      int x1 = static_cast<int>((cx - w / 2.0f) * mask_w);
      int y1 = static_cast<int>((cy - h / 2.0f) * mask_h);
      int x2 = static_cast<int>((cx + w / 2.0f) * mask_w);
      int y2 = static_cast<int>((cy + h / 2.0f) * mask_h);
      cv::rectangle(*result, cv::Point(x1, y1), cv::Point(x2, y2), color, 2);
    }

    NNDEPLOY_LOGI("Sam3SimplePostprocess: instance %d, score=%.3f", i,
                  scores_data[i]);
    kept++;
  }

  outputs_[0]->set(result, false);
  return base::kStatusCodeOk;
}

// ===================== Sam3SimpleGraphParam =====================

base::Status Sam3SimpleGraphParam::serialize(
    rapidjson::Value& json, rapidjson::Document::AllocatorType& allocator) {
  base::Status status = base::Param::serialize(json, allocator);
  CHECK_IF_ERROR_RETURN(status, "Param::serialize failed");

  json.AddMember(
      "inference_type_",
      rapidjson::Value(inference_type_.c_str(),
                       static_cast<rapidjson::SizeType>(inference_type_.size()),
                       allocator),
      allocator);
  json.AddMember(
      "device_type_",
      rapidjson::Value(device_type_.c_str(),
                       static_cast<rapidjson::SizeType>(device_type_.size()),
                       allocator),
      allocator);
  json.AddMember(
      "model_type_",
      rapidjson::Value(model_type_.c_str(),
                       static_cast<rapidjson::SizeType>(model_type_.size()),
                       allocator),
      allocator);
  json.AddMember("is_path_", is_path_, allocator);

  rapidjson::Value model_arr(rapidjson::kArrayType);
  for (const auto& v : model_value_) {
    model_arr.PushBack(
        rapidjson::Value(v.c_str(), static_cast<rapidjson::SizeType>(v.size()),
                         allocator),
        allocator);
  }
  json.AddMember("model_value_", model_arr, allocator);

  if (!text_prompt_.empty()) {
    json.AddMember(
        "text_prompt_",
        rapidjson::Value(text_prompt_.c_str(),
                         static_cast<rapidjson::SizeType>(text_prompt_.size()),
                         allocator),
        allocator);
  }

  if (!token_ids_.empty()) {
    rapidjson::Value token_arr(rapidjson::kArrayType);
    for (const auto& v : token_ids_) {
      token_arr.PushBack(static_cast<int64_t>(v), allocator);
    }
    json.AddMember("token_ids_", token_arr, allocator);
  }

  return base::kStatusCodeOk;
}

base::Status Sam3SimpleGraphParam::deserialize(rapidjson::Value& json) {
  base::Status status = base::Param::deserialize(json);
  CHECK_IF_ERROR_RETURN(status, "Param::deserialize failed");

  if (json.HasMember("inference_type_") && json["inference_type_"].IsString()) {
    inference_type_ = json["inference_type_"].GetString();
  }
  if (json.HasMember("device_type_") && json["device_type_"].IsString()) {
    device_type_ = json["device_type_"].GetString();
  }
  if (json.HasMember("model_type_") && json["model_type_"].IsString()) {
    model_type_ = json["model_type_"].GetString();
  }
  if (json.HasMember("is_path_") && json["is_path_"].IsBool()) {
    is_path_ = json["is_path_"].GetBool();
  }
  if (json.HasMember("model_value_") && json["model_value_"].IsArray()) {
    model_value_.clear();
    for (auto& v : json["model_value_"].GetArray()) {
      if (v.IsString()) {
        model_value_.push_back(v.GetString());
      }
    }
  }
  if (json.HasMember("external_model_data_") && json["external_model_data_"].IsArray()) {
    external_model_data_.clear();
    for (auto& v : json["external_model_data_"].GetArray()) {
      if (v.IsString()) {
        external_model_data_.push_back(v.GetString());
      }
    }
  }

  if (json.HasMember("text_prompt_") && json["text_prompt_"].IsString()) {
    text_prompt_ = json["text_prompt_"].GetString();
  }

  if (json.HasMember("token_ids_") && json["token_ids_"].IsArray()) {
    token_ids_.clear();
    for (auto& v : json["token_ids_"].GetArray()) {
      if (v.IsInt64()) {
        token_ids_.push_back(v.GetInt64());
      } else if (v.IsInt()) {
        token_ids_.push_back(static_cast<int64_t>(v.GetInt()));
      }
    }
  }

  // If text_prompt_ is set but no token_ids_, pre-compute defaults
  if (token_ids_.empty() && !text_prompt_.empty()) {
    // Default CLIP token IDs for "person" (most common prompt)
    // [SOS(49406), "person"(2533), EOS(49407), 0...0]
    static const int64_t default_person_tokens[] = {
        49406, 2533, 49407, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0,     0,    0,     0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
    token_ids_.assign(default_person_tokens,
                      default_person_tokens + 32);
  }

  return base::kStatusCodeOk;
}

// ===================== Sam3SimpleGraph =====================

base::Status Sam3SimpleGraph::initDynamicsGraphNodes() {
  base::Status status = base::kStatusCodeOk;

  // Preprocessing
  preprocess_node_ =
      dynamic_cast<preprocess::CvtResizePadNormTrans*>(
          this->createNode<preprocess::CvtResizePadNormTrans>("preprocess"));
  CHECK_IF_NULL_RETURN(preprocess_node_,
                       "Failed to create preprocess node");

  // Image encoder
  image_encoder_node_ =
      static_cast<Sam3SimpleImageEncoder *>(
          this->createNode<Sam3SimpleImageEncoder>("image_encoder"));
  CHECK_IF_NULL_RETURN(image_encoder_node_,
                       "Failed to create image_encoder node");

  // Language encoder (optional)
  language_encoder_node_ =
      static_cast<Sam3SimpleLanguageEncoder *>(
          this->createNode<Sam3SimpleLanguageEncoder>("language_encoder"));
  CHECK_IF_NULL_RETURN(language_encoder_node_,
                       "Failed to create language_encoder node");

  // Decoder
  decoder_node_ =
      static_cast<Sam3SimpleDecoder *>(
          this->createNode<Sam3SimpleDecoder>("decoder"));
  CHECK_IF_NULL_RETURN(decoder_node_,
                       "Failed to create decoder node");

  // Postprocess
  postprocess_node_ =
      static_cast<Sam3SimplePostprocess *>(
          this->createNode<Sam3SimplePostprocess>("postprocess"));
  CHECK_IF_NULL_RETURN(postprocess_node_,
                       "Failed to create postprocess node");

  // Create dynamic edges for decoder box prompts
  box_coords_edge_ = this->createEdge("box_coords");
  box_labels_edge_ = this->createEdge("box_labels");
  box_masks_edge_ = this->createEdge("box_masks");

  // Create dynamic edges for original image dimensions (decoder inputs [0] and [1])
  original_height_edge_ = this->createEdge("original_height");
  original_width_edge_ = this->createEdge("original_width");

  return status;
}

base::Status Sam3SimpleGraph::setInferParam(
    base::InferenceType inference_type, base::DeviceType device_type,
    base::ModelType model_type, bool is_path,
    std::vector<std::string>& model_value,
    std::vector<std::string>& external_model_data) {
  base::Status status = base::kStatusCodeOk;

  // model_value[0]: Image Encoder
  if (model_value.size() > 0 && !model_value[0].empty()) {
    status = image_encoder_node_->setInferParam(
        inference_type, device_type, model_type, is_path, model_value[0]);
    CHECK_IF_ERROR_RETURN(status, "Failed to set image encoder infer param");
    // Set external data if available
    if (external_model_data.size() > 0 && !external_model_data[0].empty()) {
      image_encoder_node_->setExternalModelData(external_model_data[0]);
    }
  }

  // model_value[1]: Decoder
  if (model_value.size() > 1 && !model_value[1].empty()) {
    status = decoder_node_->setInferParam(inference_type, device_type,
                                          model_type, is_path, model_value[1]);
    CHECK_IF_ERROR_RETURN(status, "Failed to set decoder infer param");
    // Set external data if available
    if (external_model_data.size() > 1 && !external_model_data[1].empty()) {
      decoder_node_->setExternalModelData(external_model_data[1]);
    }
  }

  // model_value[2]: Language Encoder (optional)
  if (model_value.size() > 2 && !model_value[2].empty()) {
    status = language_encoder_node_->setInferParam(
        inference_type, device_type, model_type, is_path, model_value[2]);
    CHECK_IF_ERROR_RETURN(status, "Failed to set language encoder infer param");
    // Set external data if available
    if (external_model_data.size() > 2 && !external_model_data[2].empty()) {
      language_encoder_node_->setExternalModelData(external_model_data[2]);
    }
  }

  return status;
}

base::Status Sam3SimpleGraph::defaultParam() {
  base::Status status = base::kStatusCodeOk;

  // Configure preprocessing for sam3_image_encoder.onnx input (3, 1008, 1008)
  preprocess::CvtResizePadNormTransParam *preprocess_param =
      dynamic_cast<preprocess::CvtResizePadNormTransParam *>(
          preprocess_node_->getParam());
  CHECK_IF_NULL_RETURN(preprocess_param,
                       "Failed to get preprocess param");
  preprocess_param->src_pixel_type_ = base::kPixelTypeBGR;
  preprocess_param->dst_pixel_type_ = base::kPixelTypeRGB;
  preprocess_param->interp_type_ = base::kInterpTypeLinear;
  preprocess_param->h_ = 1008;
  preprocess_param->w_ = 1008;
  preprocess_param->scale_[1] = 1.0f;
  preprocess_param->scale_[2] = 1.0f;
  preprocess_param->scale_[3] = 1.0f;
  preprocess_param->mean_[1] = 123.675;
  preprocess_param->mean_[2] = 116.28;
  preprocess_param->mean_[3] = 103.53;
  preprocess_param->std_[1] = 58.395;
  preprocess_param->std_[2] = 57.12;
  preprocess_param->std_[3] = 57.375;
  preprocess_param->normalize_ = false;
  preprocess_param->data_type_ = base::dataTypeOf<uint8_t>();

  // Configure postprocess
  Sam3PostParam *post_param =
      dynamic_cast<Sam3PostParam *>(postprocess_node_->getParam());
  if (post_param != nullptr) {
    post_param->score_threshold_ = 0.5f;
  }

  return status;
}

base::Status Sam3SimpleGraph::init() {
  // Skip base Graph::init() which tries to topologically sort nodes
  // Instead, just initialize the internal nodes directly
  base::Status status = base::kStatusCodeOk;

  // Initialize internal nodes
  if (preprocess_node_ != nullptr) {
    status = preprocess_node_->init();
    CHECK_IF_ERROR_RETURN(status, "Failed to init preprocess node");
  }

  if (image_encoder_node_ != nullptr) {
    status = image_encoder_node_->init();
    CHECK_IF_ERROR_RETURN(status, "Failed to init image_encoder node");
    infer::Infer *infer_node = image_encoder_node_->getInferNode();
    if (infer_node != nullptr) {
      status = infer_node->init();
      CHECK_IF_ERROR_RETURN(status, "Failed to init image encoder infer node");
    }
  }

  if (language_encoder_node_ != nullptr) {
    status = language_encoder_node_->init();
    CHECK_IF_ERROR_RETURN(status, "Failed to init language_encoder node");
    infer::Infer *infer_node = language_encoder_node_->getInferNode();
    if (infer_node != nullptr) {
      status = infer_node->init();
      CHECK_IF_ERROR_RETURN(status, "Failed to init language encoder infer node");
    }
  }

  if (decoder_node_ != nullptr) {
    status = decoder_node_->init();
    CHECK_IF_ERROR_RETURN(status, "Failed to init decoder node");
    infer::Infer *infer_node = decoder_node_->getInferNode();
    if (infer_node != nullptr) {
      status = infer_node->init();
      CHECK_IF_ERROR_RETURN(status, "Failed to init decoder infer node");
    }
  }

  if (postprocess_node_ != nullptr) {
    status = postprocess_node_->init();
    CHECK_IF_ERROR_RETURN(status, "Failed to init postprocess node");
  }

  initialized_ = true;
  return base::kStatusCodeOk;
}

base::Status Sam3SimpleGraph::deserialize(rapidjson::Value& json) {
  base::Status status = dag::Graph::deserialize(json);
  if (status != base::kStatusCodeOk) {
    NNDEPLOY_LOGE("Sam3SimpleGraph base deserialize failed\n");
    return status;
  }

  Sam3SimpleGraphParam* graph_param =
      dynamic_cast<Sam3SimpleGraphParam*>(param_.get());
  if (graph_param == nullptr) {
    NNDEPLOY_LOGW("Sam3SimpleGraph: param_ is not Sam3SimpleGraphParam");
    return base::kStatusCodeOk;
  }

  if (graph_param->model_value_.empty()) {
    NNDEPLOY_LOGW("Sam3SimpleGraph: no model_value_ in param");
    return base::kStatusCodeOk;
  }

  base::InferenceType inference_type =
      base::stringToInferenceType(graph_param->inference_type_);
  base::DeviceType device_type =
      base::stringToDeviceType(graph_param->device_type_);
  base::ModelType model_type =
      base::stringToModelType(graph_param->model_type_);

  status =
      this->setInferParam(inference_type, device_type, model_type,
                          graph_param->is_path_, graph_param->model_value_,
                          graph_param->external_model_data_);
  if (status != base::kStatusCodeOk) {
    NNDEPLOY_LOGE("Sam3SimpleGraph::deserialize: setInferParam failed\n");
    return status;
  }

  return base::kStatusCodeOk;
}

std::vector<dag::Edge*> Sam3SimpleGraph::forward(
    std::vector<dag::Edge*> inputs) {
  // inputs[0] = image (cv::Mat)
  // inputs[1] = text tokens (std::vector<int64_t>) [optional]
  // inputs[2..4] = box_coords, box_labels, box_masks [optional]

  // Step 1: Preprocess image
  std::vector<dag::Edge*> preproc_out = (*preprocess_node_)({inputs[0]});

  // Step 2: Image encoder
  std::vector<dag::Edge*> img_enc_out = (*image_encoder_node_)(preproc_out);
  // img_enc_out[0..5]: vision_pos_enc[0:3], backbone_fpn[0:3]

  // Step 3: Language encoder (optional)
  dag::Edge* lang_mask_edge = nullptr;
  dag::Edge* lang_features_edge = nullptr;

  // Get token_ids_ from param
  Sam3SimpleGraphParam* graph_param =
      dynamic_cast<Sam3SimpleGraphParam*>(param_.get());
  std::vector<int64_t> token_ids;
  if (graph_param != nullptr && !graph_param->token_ids_.empty()) {
    token_ids = graph_param->token_ids_;
  }

  bool has_text = (inputs.size() > 1 && inputs[1] != nullptr);
  if (has_text) {
    std::vector<dag::Edge*> lang_out = (*language_encoder_node_)({inputs[1]});
    lang_mask_edge = lang_out[0];
    lang_features_edge = lang_out[1];  // text_memory (index 1, post-transform output)
  } else if (!token_ids.empty()) {
    // Use pre-computed token IDs from param (deserialized from JSON)
    device::Device* cur_device = device::getDefaultHostDevice();
    device::TensorDesc tokens_desc;
    tokens_desc.data_type_ = base::dataTypeOf<int64_t>();
    tokens_desc.shape_ = {1, static_cast<int>(token_ids.size())};
    device::Tensor* tokens_tensor =
        new device::Tensor(cur_device, tokens_desc);
    memcpy(tokens_tensor->getData(), token_ids.data(),
           token_ids.size() * sizeof(int64_t));
    dag::Edge* token_edge = this->createEdge("text_tokens");
    token_edge->set(tokens_tensor, true);
    std::vector<dag::Edge*> lang_out =
        (*language_encoder_node_)({token_edge});
    lang_mask_edge = lang_out[0];
    lang_features_edge = lang_out[1];  // text_memory (index 1, post-transform output)
  }

  // Step 4: Prepare decoder inputs
  device::Device* cur_device = device::getDefaultHostDevice();

  // Language inputs — create zeros if no language encoder
  if (lang_mask_edge == nullptr) {
    device::TensorDesc mask_desc;
    mask_desc.data_type_ = base::dataTypeOf<bool>();
    mask_desc.shape_ = {1, 32};
    device::Tensor* zero_mask = new device::Tensor(cur_device, mask_desc);
    memset(zero_mask->getData(), 0, zero_mask->getSize());
    lang_mask_edge = this->createEdge("lang_mask_dummy");
    lang_mask_edge->set(zero_mask, true);
  }
  if (lang_features_edge == nullptr) {
    device::TensorDesc feat_desc;
    feat_desc.data_type_ = base::dataTypeOf<float>();
    feat_desc.shape_ = {32, 1, 256};
    device::Tensor* zero_feat = new device::Tensor(cur_device, feat_desc);
    memset(zero_feat->getData(), 0, zero_feat->getSize());
    lang_features_edge = this->createEdge("lang_features_dummy");
    lang_features_edge->set(zero_feat, true);
  }

  // Box prompt inputs — create defaults if not provided
  bool has_boxes = (inputs.size() > 2 && inputs[2] != nullptr);
  if (!has_boxes) {
    // Default: full-image box prompt for text-driven segmentation.
    // SAM3 decoder expects box_coords in normalized [0,1] range.
    // Use [0,0,1,1] as a "segment everything" box, letting the text
    // features guide which class/object to segment.
    device::TensorDesc coords_desc;
    coords_desc.data_type_ = base::dataTypeOf<float>();
    coords_desc.shape_ = {1, 1, 4};
    device::Tensor* full_box_coords = new device::Tensor(cur_device, coords_desc);
    float* coords_data = static_cast<float*>(full_box_coords->getData());
    coords_data[0] = 0.0f;
    coords_data[1] = 0.0f;
    coords_data[2] = 1.0f;
    coords_data[3] = 1.0f;
    box_coords_edge_->set(full_box_coords, true);

    device::TensorDesc labels_desc;
    labels_desc.data_type_ = base::dataTypeOf<int64_t>();
    labels_desc.shape_ = {1, 1};
    device::Tensor* fg_labels = new device::Tensor(cur_device, labels_desc);
    static_cast<int64_t*>(fg_labels->getData())[0] = 1;
    box_labels_edge_->set(fg_labels, true);

    device::TensorDesc masks_desc;
    masks_desc.data_type_ = base::dataTypeOf<bool>();
    masks_desc.shape_ = {1, 1};
    device::Tensor* valid_masks = new device::Tensor(cur_device, masks_desc);
    static_cast<bool*>(valid_masks->getData())[0] = true;  // valid point
    box_masks_edge_->set(valid_masks, true);
  } else {
    box_coords_edge_ = inputs[2];
    if (inputs.size() > 3 && inputs[3] != nullptr) {
      box_labels_edge_ = inputs[3];
    }
    if (inputs.size() > 4 && inputs[4] != nullptr) {
      box_masks_edge_ = inputs[4];
    }
  }

  // Step 5: Create original image dimension tensors from actual input image.
  // The decoder ONNX model expects the RAW (pre-preprocess) input image
  // dimensions so it can un-normalize box/mask predictions back to original
  // coordinate space. We extract rows (height) and cols (width) from the
  // cv::Mat on inputs[0], which the caller set before calling forward().
  device::TensorDesc dim_desc;
  dim_desc.data_type_ = base::dataTypeOf<int64_t>();
  dim_desc.shape_ = {1};

  int64_t orig_h = 1008;  // fallback: model input size
  int64_t orig_w = 1008;
  cv::Mat *input_mat = inputs[0]->getCvMat(this);
  if (input_mat != nullptr && !input_mat->empty()) {
    orig_h = static_cast<int64_t>(input_mat->rows);
    orig_w = static_cast<int64_t>(input_mat->cols);
  } else {
    NNDEPLOY_LOGI("Sam3SimpleGraph::forward(): "
                  "cannot read cv::Mat from inputs[0]; "
                  "defaulting orig_h/w to %lld/%lld\n",
                  (long long)orig_h, (long long)orig_w);
  }

  // The decoder ONNX model expects original_height/width as 0D scalar int64
  // values (not 1D [1] tensors). Passing shape [1] causes downstream Unsqueeze
  // nodes to add an extra dimension, leading to rank mismatches in Concat ops.
  // Use empty shape for true ONNX scalar semantics.
  device::TensorDesc scalar_dim_desc;
  scalar_dim_desc.data_type_ = base::dataTypeOf<int64_t>();
  // deliberately empty shape → 0D scalar
  device::Tensor *h_tensor = new device::Tensor(cur_device, scalar_dim_desc);
  memcpy(h_tensor->getData(), &orig_h, sizeof(int64_t));
  original_height_edge_->set(h_tensor, true);

  device::Tensor *w_tensor = new device::Tensor(cur_device, scalar_dim_desc);
  memcpy(w_tensor->getData(), &orig_w, sizeof(int64_t));
  original_width_edge_->set(w_tensor, true);

  // Step 6: Assemble decoder inputs (11 edges matching ONNX order)
  //   [0]  original_height (int64 scalar)
  //   [1]  original_width  (int64 scalar)
  //   [2]  vision_pos_enc_2               (img_enc_out[2])
  //   [3]  backbone_fpn_0                 (img_enc_out[3])
  //   [4]  backbone_fpn_1                 (img_enc_out[4])
  //   [5]  backbone_fpn_2                 (img_enc_out[5])
  //   [6]  language_mask                  (bool)
  //   [7]  language_features
  //   [8]  box_coords                     (1,1,4)
  //   [9]  box_labels                     (1,1)
  //   [10] box_masks                      (1,1)
  std::vector<dag::Edge*> decoder_in = {
      original_height_edge_,
      original_width_edge_,
      img_enc_out[2],  // vision_pos_enc_2
      img_enc_out[3],  // backbone_fpn_0
      img_enc_out[4],  // backbone_fpn_1
      img_enc_out[5],  // backbone_fpn_2
      lang_mask_edge,
      lang_features_edge,
      box_coords_edge_,
      box_labels_edge_,
      box_masks_edge_,
  };

  std::vector<dag::Edge*> decoder_out = (*decoder_node_)(decoder_in);
  // decoder_out[0] = boxes, [1] = scores, [2] = masks (name-based mapping)

  // Step 6: Postprocess
  std::vector<dag::Edge*> post_out =
      (*postprocess_node_)({decoder_out[0], decoder_out[1], decoder_out[2]});

  return post_out;
}

base::Status Sam3SimpleGraph::run() {
  if (!is_inner_) {
    setRunningFlag(true);
  }

  // Execute internal pipeline via forward() instead of the base Graph::run()
  // which would crash because Sam3SimpleGraph::init() skips executor creation.
  std::vector<dag::Edge*> fwd_outputs = forward(inputs_);

  // Transfer output data to the graph's output edges (wired into the parent DAG)
  for (size_t i = 0; i < fwd_outputs.size() && i < outputs_.size(); ++i) {
    // Try cv::Mat first (postprocess produces segmentation mask)
    cv::Mat* src_cv = fwd_outputs[i]->getGraphOutputCvMat();
    if (src_cv != nullptr) {
      outputs_[i]->set(new cv::Mat(src_cv->clone()), false);
      continue;
    }
    // Fall back to generic Param transfer
    base::Param* param = fwd_outputs[i]->getGraphOutputParam();
    if (param != nullptr) {
      outputs_[i]->set(param, false);
    }
  }

  if (!is_inner_) {
    setRunningFlag(false);
  }

  return base::kStatusCodeOk;
}

// ===================== Sam3LanguageEncodeNode =====================

base::Status Sam3LanguageEncodeNode::defaultParam() {
  Sam3LanguageParam* param = dynamic_cast<Sam3LanguageParam*>(param_.get());
  param->max_token_length_ = 77;
  param->hidden_dim_ = 512;
  param->num_concepts_ = 0;
  return base::kStatusCodeOk;
}

base::Status Sam3LanguageEncodeNode::setTextEncoderInferParam(
    base::InferenceType inference_type, base::DeviceType device_type,
    base::ModelType model_type, bool is_path, std::string& model_path) {
  text_encoder_param_.inference_type_ = inference_type;
  text_encoder_param_.device_type_ = device_type;
  text_encoder_param_.model_type_ = model_type;
  text_encoder_param_.is_path_ = is_path;
  text_encoder_param_.model_value_ = {model_path};

  if (text_encoder_infer_ != nullptr) {
    base::Status status = text_encoder_infer_->setInferenceType(inference_type);
    CHECK_IF_ERROR_RETURN(status, "Failed to set text encoder inference type");
    status = text_encoder_infer_->setParam(&text_encoder_param_);
    CHECK_IF_ERROR_RETURN(status, "Failed to set text encoder param");
    has_text_encoder_ = true;
  }
  return base::kStatusCodeOk;
}

base::Status Sam3LanguageEncodeNode::run() {
  Sam3LanguageParam* param = dynamic_cast<Sam3LanguageParam*>(param_.get());
  CHECK_IF_NULL_RETURN(param, "Sam3LanguageParam is null");

  Sam3ConceptParam* concept_param = nullptr;
  if (!inputs_.empty() && inputs_[0] != nullptr) {
    concept_param = inputs_[0]->get<Sam3ConceptParam>(this);
  }
  if (concept_param == nullptr || concept_param->concepts_.empty()) {
    NNDEPLOY_LOGE("No concepts provided for language encoding.");
    return base::kStatusCodeErrorInvalidValue;
  }

  int num_concepts = static_cast<int>(concept_param->concepts_.size());
  int hidden_dim = param->hidden_dim_;

  device::Device* cur_device = device::getDefaultHostDevice();

  if (has_text_encoder_ && text_encoder_infer_ != nullptr) {
    device::TensorDesc token_desc;
    token_desc.data_format_ = base::kDataFormatNCL;
    token_desc.data_type_ = base::dataTypeOf<int32_t>();
    token_desc.shape_ = {1, num_concepts, param->max_token_length_};

    device::Tensor* tokens = outputs_[0]->create(cur_device, token_desc);
    CHECK_IF_NULL_RETURN(tokens, "Failed to create token tensor");
    memset(tokens->getData(), 0, tokens->getSize());

    text_encoder_infer_->getInput(0)->set(tokens, false);
    base::Status status = text_encoder_infer_->run();
    CHECK_IF_ERROR_RETURN(status, "Text encoder inference failed");

    device::Tensor* output_tensor =
        text_encoder_infer_->getOutput(0)->getTensor(text_encoder_infer_);
    if (output_tensor != nullptr) {
      device::TensorDesc out_desc = output_tensor->getDesc();
      if (out_desc.shape_.size() >= 2) {
        hidden_dim = out_desc.shape_.back();
      }
      size_t data_size = output_tensor->getSize();
      device::Tensor* concept_tensor =
          outputs_[0]->create(cur_device, out_desc);
      CHECK_IF_NULL_RETURN(concept_tensor, "Failed to create concept tensor");
      memcpy(concept_tensor->getData(), output_tensor->getData(), data_size);

      NNDEPLOY_LOGI("SAM 3 Language Encode: %d concepts -> (%d, %d)",
                    num_concepts, num_concepts, hidden_dim);
      return base::kStatusCodeOk;
    }
  }

  // Fallback: zero embeddings
  device::TensorDesc concept_desc;
  concept_desc.data_format_ = base::kDataFormatNCL;
  concept_desc.data_type_ = base::dataTypeOf<float>();
  concept_desc.shape_ = {1, num_concepts, hidden_dim};

  device::Tensor* concept_tensor =
      outputs_[0]->create(cur_device, concept_desc);
  CHECK_IF_NULL_RETURN(concept_tensor, "Failed to create concept tensor");
  memset(concept_tensor->getData(), 0, concept_tensor->getSize());

  NNDEPLOY_LOGI("SAM 3 Language Encode (fallback): %d concepts, dim=%d",
                num_concepts, hidden_dim);

  return base::kStatusCodeOk;
}

// ===================== Sam3ConceptEncodeNode =====================

base::Status Sam3ConceptEncodeNode::run() {
  Sam3ConceptParam* param = dynamic_cast<Sam3ConceptParam*>(param_.get());
  CHECK_IF_NULL_RETURN(param, "Sam3ConceptParam is null");

  if (!inputs_.empty() && inputs_[0] != nullptr) {
    Sam3ConceptParam* input_param = inputs_[0]->get<Sam3ConceptParam>(this);
    if (input_param != nullptr) {
      param = input_param;
    }
  }

  int num_concepts = param->num_concepts_;
  int concept_dim = param->concept_dim_;
  if (num_concepts <= 0) {
    num_concepts = static_cast<int>(param->concepts_.size());
    if (num_concepts <= 0) {
      NNDEPLOY_LOGE("No concepts specified for SAM 3.");
      return base::kStatusCodeErrorInvalidValue;
    }
    param->num_concepts_ = num_concepts;
  }

  device::Device* cur_device = device::getDefaultHostDevice();

  device::TensorDesc concept_desc;
  concept_desc.data_format_ = base::kDataFormatNCL;
  concept_desc.data_type_ = base::dataTypeOf<float>();
  concept_desc.shape_ = {1, num_concepts, concept_dim};

  device::Tensor* concept_tensor =
      outputs_[0]->create(cur_device, concept_desc);
  CHECK_IF_NULL_RETURN(concept_tensor, "Failed to create concept tensor");
  memset(concept_tensor->getData(), 0, concept_tensor->getSize());

  NNDEPLOY_LOGI("SAM 3 Concept Encode: %d concepts, dim=%d", num_concepts,
                concept_dim);

  return base::kStatusCodeOk;
}

// ===================== Sam3ExemplarEncodeNode =====================

base::Status Sam3ExemplarEncodeNode::defaultParam() {
  Sam3ExemplarParam* param = dynamic_cast<Sam3ExemplarParam*>(param_.get());
  param->exemplar_images_.clear();
  param->exemplar_masks_.clear();
  param->num_exemplars_ = 0;
  return base::kStatusCodeOk;
}

base::Status Sam3ExemplarEncodeNode::setSharedEncoder(infer::Infer* pe_infer) {
  exemplar_encoder_infer_ = pe_infer;
  has_encoder_ = (pe_infer != nullptr);
  return base::kStatusCodeOk;
}

base::Status Sam3ExemplarEncodeNode::setInferParam(
    base::InferenceType inference_type, base::DeviceType device_type,
    base::ModelType model_type, bool is_path, std::string& model_path) {
  exemplar_encoder_param_.inference_type_ = inference_type;
  exemplar_encoder_param_.device_type_ = device_type;
  exemplar_encoder_param_.model_type_ = model_type;
  exemplar_encoder_param_.is_path_ = is_path;
  exemplar_encoder_param_.model_value_ = {model_path};
  return base::kStatusCodeOk;
}

static void preprocessExemplarImage(const cv::Mat& src, cv::Mat& dst) {
  cv::Mat resized;
  cv::resize(src, resized, cv::Size(1024, 1024), 0, 0, cv::INTER_LINEAR);
  cv::Mat rgb;
  if (resized.channels() == 3) {
    cv::cvtColor(resized, rgb, cv::COLOR_BGR2RGB);
  } else {
    cv::cvtColor(resized, rgb, cv::COLOR_GRAY2RGB);
  }
  rgb.convertTo(dst, CV_32FC3, 1.0 / 255.0);
  dst = (dst -
         cv::Scalar(123.675f / 255.0f, 116.28f / 255.0f, 103.53f / 255.0f)) /
        cv::Scalar(58.395f / 255.0f, 57.12f / 255.0f, 57.375f / 255.0f);
}

base::Status Sam3ExemplarEncodeNode::run() {
  Sam3ExemplarParam* param = dynamic_cast<Sam3ExemplarParam*>(param_.get());
  CHECK_IF_NULL_RETURN(param, "Sam3ExemplarParam is null");

  std::vector<cv::Mat> exemplar_images = param->exemplar_images_;
  if (!inputs_.empty() && inputs_[0] != nullptr) {
    Sam3ExemplarParam* input_param = inputs_[0]->get<Sam3ExemplarParam>(this);
    if (input_param != nullptr && !input_param->exemplar_images_.empty()) {
      exemplar_images = input_param->exemplar_images_;
    }
  }

  int num_exemplars = static_cast<int>(exemplar_images.size());
  if (num_exemplars <= 0) {
    num_exemplars = param->num_exemplars_;
    if (num_exemplars <= 0) {
      NNDEPLOY_LOGE("No exemplars specified.");
      return base::kStatusCodeErrorInvalidValue;
    }
  }

  device::Device* cur_device = device::getDefaultHostDevice();
  int exemplar_dim = 256;

  if (has_encoder_ && exemplar_encoder_infer_ != nullptr) {
    // Encode each exemplar through the shared PE
    std::vector<float> all_embeddings;
    all_embeddings.reserve(num_exemplars * exemplar_dim);

    for (int i = 0; i < num_exemplars; ++i) {
      cv::Mat exemplar_processed;
      cv::Mat exemplar_float;

      if (!exemplar_images.empty() && i < (int)exemplar_images.size()) {
        preprocessExemplarImage(exemplar_images[i], exemplar_float);
      } else {
        exemplar_float =
            cv::Mat(1024, 1024, CV_32FC3, cv::Scalar(0.0f, 0.0f, 0.0f));
      }

      // Create input tensor
      device::TensorDesc input_desc;
      input_desc.data_format_ = base::kDataFormatNCHW;
      input_desc.data_type_ = base::dataTypeOf<float>();
      input_desc.shape_ = {1, 3, 1024, 1024};
      device::Tensor* input_tensor = new device::Tensor(cur_device, input_desc);

      // Copy HWC -> CHW
      float* dst = static_cast<float*>(input_tensor->getData());
      for (int y = 0; y < 1024; ++y) {
        for (int x = 0; x < 1024; ++x) {
          cv::Vec3f pixel = exemplar_float.at<cv::Vec3f>(y, x);
          dst[0 * 1024 * 1024 + y * 1024 + x] = pixel[0];
          dst[1 * 1024 * 1024 + y * 1024 + x] = pixel[1];
          dst[2 * 1024 * 1024 + y * 1024 + x] = pixel[2];
        }
      }

      // Run through shared PE
      exemplar_encoder_infer_->getInput(0)->set(input_tensor, true);
      base::Status status = exemplar_encoder_infer_->run();
      delete input_tensor;

      if (status != base::kStatusCodeOk) {
        NNDEPLOY_LOGE("Exemplar %d PE inference failed", i);
        memset(all_embeddings.data() + all_embeddings.size(), 0,
               exemplar_dim * sizeof(float));
        all_embeddings.resize(all_embeddings.size() + exemplar_dim, 0.0f);
        continue;
      }

      // Average pool PE output
      device::Tensor* pe_output =
          exemplar_encoder_infer_->getOutput(0)->getTensor(
              exemplar_encoder_infer_);
      if (pe_output != nullptr) {
        device::TensorDesc out_desc = pe_output->getDesc();
        float* feat_data = static_cast<float*>(pe_output->getData());
        int feat_h = (out_desc.shape_.size() > 2) ? out_desc.shape_[2] : 64;
        int feat_w = (out_desc.shape_.size() > 3) ? out_desc.shape_[3] : 64;
        int feat_dim = (out_desc.shape_.size() > 1) ? out_desc.shape_[1] : 256;

        for (int d = 0; d < feat_dim; ++d) {
          float sum = 0.0f;
          for (int y = 0; y < feat_h; ++y) {
            for (int x = 0; x < feat_w; ++x) {
              sum += feat_data[d * feat_h * feat_w + y * feat_w + x];
            }
          }
          all_embeddings.push_back(sum / (feat_h * feat_w));
        }
        exemplar_dim = feat_dim;
      } else {
        all_embeddings.resize(all_embeddings.size() + exemplar_dim, 0.0f);
      }
    }

    // Create output tensor
    device::TensorDesc exemplar_desc;
    exemplar_desc.data_format_ = base::kDataFormatNCL;
    exemplar_desc.data_type_ = base::dataTypeOf<float>();
    exemplar_desc.shape_ = {1, num_exemplars, exemplar_dim};

    device::Tensor* exemplar_tensor =
        outputs_[0]->create(cur_device, exemplar_desc);
    CHECK_IF_NULL_RETURN(exemplar_tensor, "Failed to create exemplar tensor");

    float* out_data = static_cast<float*>(exemplar_tensor->getData());
    memcpy(out_data, all_embeddings.data(),
           num_exemplars * exemplar_dim * sizeof(float));

    NNDEPLOY_LOGI("SAM 3 Exemplar Encode: %d exemplars (shared PE), dim=%d",
                  num_exemplars, exemplar_dim);
    return base::kStatusCodeOk;
  }

  // Fallback: placeholder embeddings
  device::TensorDesc exemplar_desc;
  exemplar_desc.data_format_ = base::kDataFormatNCL;
  exemplar_desc.data_type_ = base::dataTypeOf<float>();
  exemplar_desc.shape_ = {1, num_exemplars, 256};

  device::Tensor* exemplar_tensor =
      outputs_[0]->create(cur_device, exemplar_desc);
  CHECK_IF_NULL_RETURN(exemplar_tensor, "Failed to create exemplar tensor");
  memset(exemplar_tensor->getData(), 0, exemplar_tensor->getSize());

  NNDEPLOY_LOGI("SAM 3 Exemplar Encode (fallback): %d exemplars",
                num_exemplars);

  return base::kStatusCodeOk;
}

// ===================== Sam3PostProcess =====================

base::Status Sam3PostProcess::defaultParam() {
  Sam3PostParam* param = dynamic_cast<Sam3PostParam*>(param_.get());
  param->score_threshold_ = 0.5f;
  param->presence_threshold_ = 0.3f;
  param->concept_threshold_ = 0.3f;
  param->model_h_ = 1024;
  param->model_w_ = 1024;
  return base::kStatusCodeOk;
}

base::Status Sam3PostProcess::run() {
  Sam3PostParam* param = dynamic_cast<Sam3PostParam*>(param_.get());
  float score_threshold = param->score_threshold_;

  device::Tensor* masks_tensor = inputs_[0]->getTensor(this);
  device::Tensor* presence_tensor = nullptr;
  device::Tensor* concept_tensor = nullptr;

  if (inputs_.size() > 1 && inputs_[1] != nullptr) {
    presence_tensor = inputs_[1]->getTensor(this);
  }
  if (inputs_.size() > 2 && inputs_[2] != nullptr) {
    concept_tensor = inputs_[2]->getTensor(this);
  }

  CHECK_IF_NULL_RETURN(masks_tensor, "Masks tensor is null");

  device::TensorDesc masks_desc = masks_tensor->getDesc();
  int result_num = masks_desc.shape_[1];
  int height = masks_desc.shape_[2];
  int width = masks_desc.shape_[3];

  float* masks_data = static_cast<float*>(masks_tensor->getData());
  float* presence_data = nullptr;
  if (presence_tensor != nullptr) {
    presence_data = static_cast<float*>(presence_tensor->getData());
  }

  // Per-concept matching from concept matcher
  float* concept_data = nullptr;
  int num_concepts = 1;
  if (concept_tensor != nullptr) {
    concept_data = static_cast<float*>(concept_tensor->getData());
    device::TensorDesc aux_desc = concept_tensor->getDesc();
    if (aux_desc.shape_.size() >= 3) {
      num_concepts = aux_desc.shape_.back();
    }
  }

  // Build instance list from presence filtering + concept matching
  std::vector<DetectionInstance> instances;
  for (int i = 0; i < result_num; ++i) {
    float presence = (presence_data != nullptr) ? presence_data[i] : 1.0f;

    if (presence < param->presence_threshold_) {
      continue;
    }

    // Find best concept match
    int best_concept = 0;
    float best_concept_score = 1.0f;
    if (concept_data != nullptr) {
      best_concept_score = 0.0f;
      for (int c = 0; c < num_concepts; ++c) {
        float score = concept_data[i * num_concepts + c];
        if (score > best_concept_score) {
          best_concept_score = score;
          best_concept = c;
        }
      }
      if (best_concept_score < param->concept_threshold_) {
        continue;
      }
    }

    DetectionInstance inst;
    inst.concept_idx = best_concept;
    inst.presence_score = presence;
    inst.concept_score = best_concept_score;
    inst.final_score = presence * best_concept_score;

    // Extract mask
    cv::Mat mask(height, width, CV_32FC1, masks_data + i * height * width);
    inst.mask = mask.clone();

    // Compute bounding box from mask
    cv::Mat mask_8u;
    mask.convertTo(mask_8u, CV_8UC1, 255.0f);
    cv::findContours(mask_8u, std::vector<std::vector<cv::Point>>(),
                     cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);
    cv::Moments m = cv::moments(mask_8u, true);
    if (m.m00 > 0) {
      // Simple bbox from non-zero pixels
      cv::Mat non_zero;
      cv::findNonZero(mask_8u, non_zero);
      if (!non_zero.empty()) {
        inst.bbox = cv::boundingRect(non_zero);
      }
    }

    if (inst.final_score >= score_threshold) {
      instances.push_back(inst);
    }
  }

  // Create visualization
  cv::Mat* result = new cv::Mat(height, width, CV_8UC3, cv::Scalar(0, 0, 0));
  std::vector<cv::Scalar> colors = {
      cv::Scalar(255, 0, 0),   cv::Scalar(0, 255, 0),   cv::Scalar(0, 0, 255),
      cv::Scalar(255, 255, 0), cv::Scalar(255, 0, 255), cv::Scalar(0, 255, 255),
      cv::Scalar(128, 0, 0),   cv::Scalar(0, 128, 0),   cv::Scalar(0, 0, 128),
      cv::Scalar(128, 128, 0),
  };

  for (size_t i = 0; i < instances.size(); ++i) {
    auto& inst = instances[i];
    cv::Scalar color = colors[inst.concept_idx % colors.size()];

    cv::Mat mask_8u;
    inst.mask.convertTo(mask_8u, CV_8UC1, 255.0f);
    for (int y = 0; y < height; ++y) {
      for (int x = 0; x < width; ++x) {
        if (mask_8u.at<uint8_t>(y, x) > 127) {
          result->at<cv::Vec3b>(y, x) = cv::Vec3b(
              static_cast<uint8_t>(color[0]), static_cast<uint8_t>(color[1]),
              static_cast<uint8_t>(color[2]));
        }
      }
    }

    NNDEPLOY_LOGI(
        "SAM 3 PostProcess: inst %zu, concept %d, "
        "pres=%.3f, conc=%.3f, final=%.3f",
        i, inst.concept_idx, inst.presence_score, inst.concept_score,
        inst.final_score);
  }

  outputs_[0]->set(result, false);
  return base::kStatusCodeOk;
}

// ===================== NEW: Sam3PerceptionEncoder =====================

base::Status Sam3PerceptionEncoder::defaultParam() {
  Sam3PerceptionEncoderParam* param =
      dynamic_cast<Sam3PerceptionEncoderParam*>(param_.get());
  param->embed_dim_ = 256;
  param->image_size_ = 1024;
  param->use_high_res_ = false;
  return base::kStatusCodeOk;
}

base::Status Sam3PerceptionEncoder::setInferParam(
    base::InferenceType inference_type, base::DeviceType device_type,
    base::ModelType model_type, bool is_path, std::string& model_path) {
  pe_infer_param_.inference_type_ = inference_type;
  pe_infer_param_.device_type_ = device_type;
  pe_infer_param_.model_type_ = model_type;
  pe_infer_param_.is_path_ = is_path;
  pe_infer_param_.model_value_ = {model_path};

  if (pe_infer_node_ == nullptr) {
    pe_infer_node_ = new infer::Infer("pe_infer");
  }
  base::Status status = pe_infer_node_->setInferenceType(inference_type);
  CHECK_IF_ERROR_RETURN(status, "Failed to set PE inference type");
  status = pe_infer_node_->setParam(&pe_infer_param_);
  CHECK_IF_ERROR_RETURN(status, "Failed to set PE param");
  return status;
}

base::Status Sam3PerceptionEncoder::run() {
  if (pe_infer_node_ == nullptr) {
    NNDEPLOY_LOGE("PerceptionEncoder: no infer node configured.");
    return base::kStatusCodeErrorInvalidValue;
  }

  // Pass input tensor through to the PE model
  device::Tensor* input_tensor = inputs_[0]->getTensor(this);
  CHECK_IF_NULL_RETURN(input_tensor, "PE input tensor is null");

  pe_infer_node_->getInput(0)->set(input_tensor, false);
  base::Status status = pe_infer_node_->run();
  CHECK_IF_ERROR_RETURN(status, "PE inference failed");

  // Copy output to our output edge
  device::Tensor* output_tensor =
      pe_infer_node_->getOutput(0)->getTensor(pe_infer_node_);
  CHECK_IF_NULL_RETURN(output_tensor, "PE output tensor is null");

  device::Device* cur_device = device::getDefaultHostDevice();
  device::TensorDesc out_desc = output_tensor->getDesc();
  device::Tensor* out = outputs_[0]->create(cur_device, out_desc);
  CHECK_IF_NULL_RETURN(out, "Failed to create PE output");
  memcpy(out->getData(), output_tensor->getData(), output_tensor->getSize());

  return base::kStatusCodeOk;
}

// ===================== NEW: Sam3DetectorDecoder =====================

base::Status Sam3DetectorDecoder::defaultParam() {
  Sam3DetectorDecoderParam* param =
      dynamic_cast<Sam3DetectorDecoderParam*>(param_.get());
  param->num_queries_ = 200;
  param->query_dim_ = 256;
  param->has_presence_token_ = true;
  param->num_decoder_layers_ = 6;
  param->box_refine_ = true;
  return base::kStatusCodeOk;
}

base::Status Sam3DetectorDecoder::setInferParam(
    base::InferenceType inference_type, base::DeviceType device_type,
    base::ModelType model_type, bool is_path, std::string& model_path) {
  detector_infer_param_.inference_type_ = inference_type;
  detector_infer_param_.device_type_ = device_type;
  detector_infer_param_.model_type_ = model_type;
  detector_infer_param_.is_path_ = is_path;
  detector_infer_param_.model_value_ = {model_path};
  detector_infer_param_.is_dynamic_shape_ = true;
  detector_infer_param_.max_shape_.insert(
      {"image_embeddings", {1, 256, 64, 64}});
  detector_infer_param_.max_shape_.insert({"prompt_embeddings", {1, 255, 512}});

  if (detector_infer_node_ == nullptr) {
    detector_infer_node_ = new infer::Infer("detector_infer");
  }
  base::Status status = detector_infer_node_->setInferenceType(inference_type);
  CHECK_IF_ERROR_RETURN(status, "Failed to set detector inference type");
  status = detector_infer_node_->setParam(&detector_infer_param_);
  CHECK_IF_ERROR_RETURN(status, "Failed to set detector param");
  return status;
}

base::Status Sam3DetectorDecoder::run() {
  if (detector_infer_node_ == nullptr) {
    NNDEPLOY_LOGE("DetectorDecoder: no infer node configured.");
    return base::kStatusCodeErrorInvalidValue;
  }

  Sam3DetectorDecoderParam* param =
      dynamic_cast<Sam3DetectorDecoderParam*>(param_.get());

  device::Tensor* image_features = inputs_[0]->getTensor(this);
  CHECK_IF_NULL_RETURN(image_features, "Detector: image features is null");

  device::Tensor* concept_embeddings = nullptr;
  if (inputs_.size() > 1 && inputs_[1] != nullptr) {
    concept_embeddings = inputs_[1]->getTensor(this);
  }

  detector_infer_node_->getInput(0)->set(image_features, false);
  if (concept_embeddings != nullptr) {
    detector_infer_node_->getInput(1)->set(concept_embeddings, false);
  }

  base::Status status = detector_infer_node_->run();
  CHECK_IF_ERROR_RETURN(status, "Detector inference failed");

  // Map outputs
  auto* out0 =
      detector_infer_node_->getOutput(0)->getTensor(detector_infer_node_);
  auto* out1 =
      detector_infer_node_->getOutput(1)->getTensor(detector_infer_node_);
  auto* out2 =
      detector_infer_node_->getOutput(2)->getTensor(detector_infer_node_);
  std::vector<device::Tensor*> out_tensors = {out0, out1, out2};
  if (detector_infer_node_->getOutputCount() > 3) {
    out_tensors.push_back(
        detector_infer_node_->getOutput(3)->getTensor(detector_infer_node_));
  }
  if (out_tensors.size() < 3) {
    NNDEPLOY_LOGE("DetectorDecoder: expected >=3 outputs, got %zu",
                  out_tensors.size());
    return base::kStatusCodeErrorInvalidValue;
  }

  device::Device* cur_device = device::getDefaultHostDevice();

  // Output[0]: masks
  device::TensorDesc mask_desc = out_tensors[0]->getDesc();
  device::Tensor* masks_out = outputs_[0]->create(cur_device, mask_desc);
  CHECK_IF_NULL_RETURN(masks_out, "Failed to create masks output");
  memcpy(masks_out->getData(), out_tensors[0]->getData(),
         out_tensors[0]->getSize());

  // Output[1]: presence scores
  device::TensorDesc pres_desc = out_tensors[1]->getDesc();
  device::Tensor* pres_out = outputs_[1]->create(cur_device, pres_desc);
  CHECK_IF_NULL_RETURN(pres_out, "Failed to create presence output");
  memcpy(pres_out->getData(), out_tensors[1]->getData(),
         out_tensors[1]->getSize());

  // Output[2]: query embeddings (for concept matching)
  device::TensorDesc query_desc = out_tensors[2]->getDesc();
  device::Tensor* query_out = outputs_[2]->create(cur_device, query_desc);
  CHECK_IF_NULL_RETURN(query_out, "Failed to create query output");
  memcpy(query_out->getData(), out_tensors[2]->getData(),
         out_tensors[2]->getSize());

  // Optional Output[3]: box predictions
  if (outputs_.size() > 3 && outputs_[3] != nullptr && out_tensors.size() > 3) {
    device::TensorDesc box_desc = out_tensors[3]->getDesc();
    device::Tensor* box_out = outputs_[3]->create(cur_device, box_desc);
    if (box_out != nullptr) {
      memcpy(box_out->getData(), out_tensors[3]->getData(),
             out_tensors[3]->getSize());
    }
  }

  NNDEPLOY_LOGI("Sam3DetectorDecoder: %d queries, masks %dx%d",
                param->num_queries_, mask_desc.shape_[2], mask_desc.shape_[3]);

  return base::kStatusCodeOk;
}

// ===================== NEW: Sam3PresenceHead =====================

base::Status Sam3PresenceHead::defaultParam() {
  Sam3PresenceHeadParam* param =
      dynamic_cast<Sam3PresenceHeadParam*>(param_.get());
  param->presence_threshold_ = 0.3f;
  param->enable_nms_ = true;
  param->nms_threshold_ = 0.7f;
  return base::kStatusCodeOk;
}

float Sam3PresenceHead::computeMaskIOU(const float* mask_a, const float* mask_b,
                                       int h, int w) {
  float intersection = 0.0f;
  float union_area = 0.0f;
  for (int i = 0; i < h * w; ++i) {
    bool a = mask_a[i] > 0.5f;
    bool b = mask_b[i] > 0.5f;
    if (a && b) intersection += 1.0f;
    if (a || b) union_area += 1.0f;
  }
  return (union_area > 0.0f) ? intersection / union_area : 0.0f;
}

base::Status Sam3PresenceHead::run() {
  Sam3PresenceHeadParam* param =
      dynamic_cast<Sam3PresenceHeadParam*>(param_.get());

  device::Tensor* presence_scores = inputs_[0]->getTensor(this);
  device::Tensor* masks_tensor = nullptr;
  device::Tensor* query_embeddings = nullptr;

  CHECK_IF_NULL_RETURN(presence_scores, "PresenceHead: scores is null");

  if (inputs_.size() > 1 && inputs_[1] != nullptr) {
    masks_tensor = inputs_[1]->getTensor(this);
  }
  if (inputs_.size() > 2 && inputs_[2] != nullptr) {
    query_embeddings = inputs_[2]->getTensor(this);
  }

  device::TensorDesc pres_desc = presence_scores->getDesc();
  int num_queries = pres_desc.shape_[1];

  float* pres_data = static_cast<float*>(presence_scores->getData());
  float* mask_data = nullptr;
  int mask_h = 0, mask_w = 0;
  if (masks_tensor != nullptr) {
    device::TensorDesc mdesc = masks_tensor->getDesc();
    mask_data = static_cast<float*>(masks_tensor->getData());
    mask_h = mdesc.shape_[2];
    mask_w = mdesc.shape_[3];
  }

  // Step 1: Filter by presence threshold
  std::vector<int> kept_indices;
  for (int i = 0; i < num_queries; ++i) {
    if (pres_data[i] >= param->presence_threshold_) {
      kept_indices.push_back(i);
    }
  }

  // Step 2: Apply NMS on kept queries
  if (param->enable_nms_ && mask_data != nullptr && !kept_indices.empty()) {
    // Sort by presence score descending
    std::sort(
        kept_indices.begin(), kept_indices.end(),
        [&pres_data](int a, int b) { return pres_data[a] > pres_data[b]; });

    std::vector<int> nms_kept;
    for (size_t i = 0; i < kept_indices.size(); ++i) {
      int idx_i = kept_indices[i];
      bool keep = true;
      for (size_t j = 0; j < nms_kept.size(); ++j) {
        int idx_j = nms_kept[j];
        float iou =
            computeMaskIOU(mask_data + idx_i * mask_h * mask_w,
                           mask_data + idx_j * mask_h * mask_w, mask_h, mask_w);
        if (iou > param->nms_threshold_) {
          keep = false;
          break;
        }
      }
      if (keep) {
        nms_kept.push_back(idx_i);
      }
    }
    kept_indices = nms_kept;
  }

  // Step 3: Build output tensors
  int num_filtered = static_cast<int>(kept_indices.size());
  device::Device* cur_device = device::getDefaultHostDevice();

  // Output[0]: Filtered query indices (int32)
  device::TensorDesc idx_desc;
  idx_desc.data_format_ = base::kDataFormatNCL;
  idx_desc.data_type_ = base::dataTypeOf<int32_t>();
  idx_desc.shape_ = {1, num_filtered, 1};
  device::Tensor* idx_out = outputs_[0]->create(cur_device, idx_desc);
  CHECK_IF_NULL_RETURN(idx_out, "Failed to create indices output");
  int32_t* idx_data = static_cast<int32_t*>(idx_out->getData());
  for (int i = 0; i < num_filtered; ++i) {
    idx_data[i] = kept_indices[i];
  }

  // Output[1]: Filtered masks
  if (outputs_.size() > 1 && outputs_[1] != nullptr &&
      masks_tensor != nullptr) {
    device::TensorDesc mask_desc = masks_tensor->getDesc();
    mask_desc.shape_[1] = num_filtered;
    device::Tensor* mask_out = outputs_[1]->create(cur_device, mask_desc);
    CHECK_IF_NULL_RETURN(mask_out, "Failed to create masks output");
    float* dst = static_cast<float*>(mask_out->getData());
    int stride = mask_h * mask_w;
    for (int i = 0; i < num_filtered; ++i) {
      memcpy(dst + i * stride, mask_data + kept_indices[i] * stride,
             stride * sizeof(float));
    }
  }

  // Output[2]: Filtered query embeddings
  if (outputs_.size() > 2 && outputs_[2] != nullptr &&
      query_embeddings != nullptr) {
    device::TensorDesc qdesc = query_embeddings->getDesc();
    qdesc.shape_[1] = num_filtered;
    int emb_dim = qdesc.shape_[2];
    device::Tensor* qout = outputs_[2]->create(cur_device, qdesc);
    CHECK_IF_NULL_RETURN(qout, "Failed to create query output");
    float* dst = static_cast<float*>(qout->getData());
    float* src = static_cast<float*>(query_embeddings->getData());
    for (int i = 0; i < num_filtered; ++i) {
      memcpy(dst + i * emb_dim, src + kept_indices[i] * emb_dim,
             emb_dim * sizeof(float));
    }
  }

  // Output[3]: Filtered presence scores
  if (outputs_.size() > 3 && outputs_[3] != nullptr) {
    device::TensorDesc score_desc;
    score_desc.data_format_ = base::kDataFormatNCL;
    score_desc.data_type_ = base::dataTypeOf<float>();
    score_desc.shape_ = {1, num_filtered, 1};
    device::Tensor* sout = outputs_[3]->create(cur_device, score_desc);
    CHECK_IF_NULL_RETURN(sout, "Failed to create scores output");
    float* dst = static_cast<float*>(sout->getData());
    for (int i = 0; i < num_filtered; ++i) {
      dst[i] = pres_data[kept_indices[i]];
    }
  }

  NNDEPLOY_LOGI("Sam3PresenceHead: %d queries -> %d after filtering + NMS",
                num_queries, num_filtered);

  return base::kStatusCodeOk;
}

// ===================== NEW: Sam3ConceptMatcher =====================

base::Status Sam3ConceptMatcher::defaultParam() {
  Sam3ConceptMatcherParam* param =
      dynamic_cast<Sam3ConceptMatcherParam*>(param_.get());
  param->similarity_threshold_ = 0.2f;
  param->scoring_type_ = "dot_product";
  param->per_concept_nms_ = true;
  return base::kStatusCodeOk;
}

base::Status Sam3ConceptMatcher::run() {
  Sam3ConceptMatcherParam* param =
      dynamic_cast<Sam3ConceptMatcherParam*>(param_.get());

  device::Tensor* query_embeddings = inputs_[0]->getTensor(this);
  device::Tensor* concept_embeddings = inputs_[1]->getTensor(this);

  CHECK_IF_NULL_RETURN(query_embeddings, "ConceptMatcher: queries is null");
  CHECK_IF_NULL_RETURN(concept_embeddings, "ConceptMatcher: concepts is null");

  device::TensorDesc qdesc = query_embeddings->getDesc();
  device::TensorDesc cdesc = concept_embeddings->getDesc();

  int num_queries = qdesc.shape_[1];
  int num_concepts = cdesc.shape_[1];
  int query_dim = qdesc.shape_[2];
  int concept_dim = cdesc.shape_[2];

  float* query_data = static_cast<float*>(query_embeddings->getData());
  float* concept_data = static_cast<float*>(concept_embeddings->getData());

  // Compute similarity matrix: query_embeddings @ concept_embeddings^T
  int sim_dim = std::min(query_dim, concept_dim);

  // Allocate similarity on host
  std::vector<float> similarity(num_queries * num_concepts, 0.0f);

  for (int q = 0; q < num_queries; ++q) {
    for (int c = 0; c < num_concepts; ++c) {
      float dot = 0.0f;
      for (int d = 0; d < sim_dim; ++d) {
        dot +=
            query_data[q * query_dim + d] * concept_data[c * concept_dim + d];
      }
      if (param->scoring_type_ == "cosine") {
        float qnorm = 0.0f, cnorm = 0.0f;
        for (int d = 0; d < sim_dim; ++d) {
          qnorm +=
              query_data[q * query_dim + d] * query_data[q * query_dim + d];
          cnorm += concept_data[c * concept_dim + d] *
                   concept_data[c * concept_dim + d];
        }
        qnorm = std::sqrt(qnorm + 1e-8f);
        cnorm = std::sqrt(cnorm + 1e-8f);
        dot = dot / (qnorm * cnorm);
      }
      similarity[q * num_concepts + c] = dot;
    }
  }

  // Softmax over concepts for each query
  std::vector<float> concept_scores(num_queries * num_concepts, 0.0f);
  std::vector<float> final_scores(num_queries, 0.0f);
  std::vector<int> best_concepts(num_queries, 0);

  for (int q = 0; q < num_queries; ++q) {
    float max_val = -1e10f;
    for (int c = 0; c < num_concepts; ++c) {
      if (similarity[q * num_concepts + c] > max_val) {
        max_val = similarity[q * num_concepts + c];
      }
    }
    // Apply threshold
    bool any_above_threshold = false;
    for (int c = 0; c < num_concepts; ++c) {
      float s = similarity[q * num_concepts + c] - max_val;
      float exp_s = std::exp(s);
      concept_scores[q * num_concepts + c] = exp_s;
      if (similarity[q * num_concepts + c] >= param->similarity_threshold_) {
        any_above_threshold = true;
      }
    }
    // Softmax normalization
    float sum_exp = 0.0f;
    for (int c = 0; c < num_concepts; ++c) {
      sum_exp += concept_scores[q * num_concepts + c];
    }
    if (sum_exp > 0.0f) {
      for (int c = 0; c < num_concepts; ++c) {
        concept_scores[q * num_concepts + c] /= sum_exp;
      }
    }

    // Best concept
    int best_c = 0;
    float best_s = concept_scores[q * num_concepts];
    for (int c = 1; c < num_concepts; ++c) {
      if (concept_scores[q * num_concepts + c] > best_s) {
        best_s = concept_scores[q * num_concepts + c];
        best_c = c;
      }
    }
    best_concepts[q] = best_c;
    final_scores[q] = best_s;
  }

  // Write output tensors
  device::Device* cur_device = device::getDefaultHostDevice();

  // Output[0]: concept scores (num_queries, num_concepts)
  device::TensorDesc cs_desc;
  cs_desc.data_format_ = base::kDataFormatNCL;
  cs_desc.data_type_ = base::dataTypeOf<float>();
  cs_desc.shape_ = {1, num_queries, num_concepts};
  device::Tensor* cs_out = outputs_[0]->create(cur_device, cs_desc);
  CHECK_IF_NULL_RETURN(cs_out, "Failed to create concept scores output");
  memcpy(cs_out->getData(), concept_scores.data(),
         num_queries * num_concepts * sizeof(float));

  // Output[1]: final scores (num_queries)
  device::TensorDesc fs_desc;
  fs_desc.data_format_ = base::kDataFormatNCL;
  fs_desc.data_type_ = base::dataTypeOf<float>();
  fs_desc.shape_ = {1, num_queries, 1};
  device::Tensor* fs_out = outputs_[1]->create(cur_device, fs_desc);
  CHECK_IF_NULL_RETURN(fs_out, "Failed to create final scores output");
  memcpy(fs_out->getData(), final_scores.data(), num_queries * sizeof(float));

  // Output[2]: best concept indices (num_queries)
  device::TensorDesc bc_desc;
  bc_desc.data_format_ = base::kDataFormatNCL;
  bc_desc.data_type_ = base::dataTypeOf<int32_t>();
  bc_desc.shape_ = {1, num_queries, 1};
  device::Tensor* bc_out = outputs_[2]->create(cur_device, bc_desc);
  CHECK_IF_NULL_RETURN(bc_out, "Failed to create best concept output");
  int32_t* bc_data = static_cast<int32_t*>(bc_out->getData());
  for (int q = 0; q < num_queries; ++q) {
    bc_data[q] = best_concepts[q];
  }

  NNDEPLOY_LOGI("Sam3ConceptMatcher: %d queries x %d concepts", num_queries,
                num_concepts);

  return base::kStatusCodeOk;
}

// ===================== NEW: Sam3MemoryEncoder =====================

base::Status Sam3MemoryEncoder::defaultParam() {
  Sam3MemoryEncoderParam* param =
      dynamic_cast<Sam3MemoryEncoderParam*>(param_.get());
  param->memory_dim_ = 256;
  param->max_memory_frames_ = 16;
  return base::kStatusCodeOk;
}

base::Status Sam3MemoryEncoder::setInferParam(
    base::InferenceType inference_type, base::DeviceType device_type,
    base::ModelType model_type, bool is_path, std::string& model_path) {
  memory_infer_param_.inference_type_ = inference_type;
  memory_infer_param_.device_type_ = device_type;
  memory_infer_param_.model_type_ = model_type;
  memory_infer_param_.is_path_ = is_path;
  memory_infer_param_.model_value_ = {model_path};

  if (memory_infer_node_ == nullptr) {
    memory_infer_node_ = new infer::Infer("memory_encoder");
  }
  base::Status status = memory_infer_node_->setInferenceType(inference_type);
  CHECK_IF_ERROR_RETURN(status, "Failed to set memory encoder type");
  status = memory_infer_node_->setParam(&memory_infer_param_);
  CHECK_IF_ERROR_RETURN(status, "Failed to set memory encoder param");
  return status;
}

base::Status Sam3MemoryEncoder::run() {
  if (memory_infer_node_ == nullptr) {
    NNDEPLOY_LOGE("MemoryEncoder: no infer node configured.");
    return base::kStatusCodeErrorInvalidValue;
  }

  device::Tensor* image_features = inputs_[0]->getTensor(this);
  device::Tensor* masks = inputs_[1]->getTensor(this);

  CHECK_IF_NULL_RETURN(image_features, "MemoryEncoder: image features null");
  CHECK_IF_NULL_RETURN(masks, "MemoryEncoder: masks null");

  memory_infer_node_->getInput(0)->set(image_features, false);
  memory_infer_node_->getInput(1)->set(masks, false);
  base::Status status = memory_infer_node_->run();
  CHECK_IF_ERROR_RETURN(status, "Memory encoder inference failed");

  device::Tensor* output =
      memory_infer_node_->getOutput(0)->getTensor(memory_infer_node_);
  CHECK_IF_NULL_RETURN(output, "MemoryEncoder output null");

  device::Device* cur_device = device::getDefaultHostDevice();
  device::TensorDesc out_desc = output->getDesc();
  device::Tensor* out = outputs_[0]->create(cur_device, out_desc);
  CHECK_IF_NULL_RETURN(out, "Failed to create mem encoder output");
  memcpy(out->getData(), output->getData(), output->getSize());

  return base::kStatusCodeOk;
}

// ===================== NEW: Sam3TrackerMaskDecoder =====================

base::Status Sam3TrackerMaskDecoder::defaultParam() {
  Sam3TrackerMaskDecoderParam* param =
      dynamic_cast<Sam3TrackerMaskDecoderParam*>(param_.get());
  param->num_mask_embeddings_ = 4;
  param->embedding_dim_ = 256;
  return base::kStatusCodeOk;
}

base::Status Sam3TrackerMaskDecoder::setInferParam(
    base::InferenceType inference_type, base::DeviceType device_type,
    base::ModelType model_type, bool is_path, std::string& model_path) {
  tracker_infer_param_.inference_type_ = inference_type;
  tracker_infer_param_.device_type_ = device_type;
  tracker_infer_param_.model_type_ = model_type;
  tracker_infer_param_.is_path_ = is_path;
  tracker_infer_param_.model_value_ = {model_path};
  tracker_infer_param_.is_dynamic_shape_ = true;

  if (tracker_infer_node_ == nullptr) {
    tracker_infer_node_ = new infer::Infer("tracker_decoder");
  }
  base::Status status = tracker_infer_node_->setInferenceType(inference_type);
  CHECK_IF_ERROR_RETURN(status, "Failed to set tracker type");
  status = tracker_infer_node_->setParam(&tracker_infer_param_);
  CHECK_IF_ERROR_RETURN(status, "Failed to set tracker param");
  return status;
}

base::Status Sam3TrackerMaskDecoder::run() {
  if (tracker_infer_node_ == nullptr) {
    NNDEPLOY_LOGE("TrackerMaskDecoder: no infer node configured.");
    return base::kStatusCodeErrorInvalidValue;
  }

  device::Tensor* image_features = inputs_[0]->getTensor(this);
  device::Tensor* memory_bank = inputs_[1]->getTensor(this);
  device::Tensor* object_queries = nullptr;

  CHECK_IF_NULL_RETURN(image_features, "Tracker: image features null");
  CHECK_IF_NULL_RETURN(memory_bank, "Tracker: memory bank null");

  if (inputs_.size() > 2 && inputs_[2] != nullptr) {
    object_queries = inputs_[2]->getTensor(this);
  }

  tracker_infer_node_->getInput(0)->set(image_features, false);
  tracker_infer_node_->getInput(1)->set(memory_bank, false);
  if (object_queries != nullptr) {
    tracker_infer_node_->getInput(2)->set(object_queries, false);
  }

  base::Status status = tracker_infer_node_->run();
  CHECK_IF_ERROR_RETURN(status, "Tracker inference failed");

  auto* track_out0 =
      tracker_infer_node_->getOutput(0)->getTensor(tracker_infer_node_);
  auto* track_out1 =
      tracker_infer_node_->getOutput(1)->getTensor(tracker_infer_node_);
  std::vector<device::Tensor*> out_tensors = {track_out0, track_out1};

  device::Device* cur_device = device::getDefaultHostDevice();

  // Output[0]: refined masks
  if (outputs_.size() > 0 && outputs_[0] != nullptr && out_tensors.size() > 0) {
    device::TensorDesc mask_desc = out_tensors[0]->getDesc();
    device::Tensor* mask_out = outputs_[0]->create(cur_device, mask_desc);
    CHECK_IF_NULL_RETURN(mask_out, "Failed to create tracker mask output");
    memcpy(mask_out->getData(), out_tensors[0]->getData(),
           out_tensors[0]->getSize());
  }

  // Output[1]: per-instance scores
  if (outputs_.size() > 1 && outputs_[1] != nullptr && out_tensors.size() > 1) {
    device::TensorDesc score_desc = out_tensors[1]->getDesc();
    device::Tensor* score_out = outputs_[1]->create(cur_device, score_desc);
    CHECK_IF_NULL_RETURN(score_out, "Failed to create tracker score output");
    memcpy(score_out->getData(), out_tensors[1]->getData(),
           out_tensors[1]->getSize());
  }

  return base::kStatusCodeOk;
}

// ===================== NEW: Sam3MemoryManager =====================

base::Status Sam3MemoryManager::defaultParam() {
  Sam3MemoryManagerParam* param =
      dynamic_cast<Sam3MemoryManagerParam*>(param_.get());
  param->max_memory_frames_ = 16;
  param->frame_stride_ = 1;
  param->enable_memory_temperature_ = false;
  return base::kStatusCodeOk;
}

base::Status Sam3MemoryManager::run() {
  Sam3MemoryManagerParam* param =
      dynamic_cast<Sam3MemoryManagerParam*>(param_.get());

  device::Tensor* new_memory = inputs_[0]->getTensor(this);
  CHECK_IF_NULL_RETURN(new_memory, "MemoryManager: new memory tensor null");

  device::TensorDesc mem_desc = new_memory->getDesc();
  int memory_dim = mem_desc.shape_[2];

  // Clone the new memory tensor and add to bank
  device::Device* cur_device = device::getDefaultHostDevice();
  device::TensorDesc clone_desc = mem_desc;
  device::Tensor* cloned = new device::Tensor(cur_device, clone_desc);
  if (cloned != nullptr) {
    memcpy(cloned->getData(), new_memory->getData(), new_memory->getSize());
    memory_bank_.add(cloned, current_frame_);
  }

  current_frame_++;

  // Concatenate all memory tensors in bank for output
  size_t bank_size = memory_bank_.size();
  if (bank_size == 0) {
    NNDEPLOY_LOGE("MemoryManager: empty memory bank");
    return base::kStatusCodeOk;
  }

  // Output[0]: concatenated memory (1, bank_size * memory_dim)
  int total_dim = static_cast<int>(bank_size) * memory_dim;
  device::TensorDesc cat_desc;
  cat_desc.data_format_ = base::kDataFormatNCL;
  cat_desc.data_type_ = base::dataTypeOf<float>();
  cat_desc.shape_ = {1, 1, total_dim};
  device::Tensor* cat_out = outputs_[0]->create(cur_device, cat_desc);
  CHECK_IF_NULL_RETURN(cat_out, "Failed to create concatenated output");

  float* dst = static_cast<float*>(cat_out->getData());
  for (size_t i = 0; i < bank_size; ++i) {
    if (memory_bank_.memory_tensors_[i] != nullptr) {
      float* src =
          static_cast<float*>(memory_bank_.memory_tensors_[i]->getData());
      memcpy(dst + i * memory_dim, src, memory_dim * sizeof(float));
    }
  }

  // Output[1]: valid mask (1, bank_size)
  device::TensorDesc valid_desc;
  valid_desc.data_format_ = base::kDataFormatNCL;
  valid_desc.data_type_ = base::dataTypeOf<float>();
  valid_desc.shape_ = {1, static_cast<int>(bank_size), 1};
  device::Tensor* valid_out = outputs_[1]->create(cur_device, valid_desc);
  CHECK_IF_NULL_RETURN(valid_out, "Failed to create valid mask output");
  float* valid_data = static_cast<float*>(valid_out->getData());
  for (size_t i = 0; i < bank_size; ++i) {
    valid_data[i] = 1.0f;
  }

  NNDEPLOY_LOGI("Sam3MemoryManager: bank size = %zu, dim = %d", bank_size,
                memory_dim);

  return base::kStatusCodeOk;
}

// ===================== Sam3GraphParam =====================

base::Status Sam3GraphParam::serialize(
    rapidjson::Value& json, rapidjson::Document::AllocatorType& allocator) {
  json.SetObject();
  json.AddMember("inference_type_",
                 rapidjson::Value(inference_type_.c_str(), allocator),
                 allocator);
  json.AddMember("device_type_",
                 rapidjson::Value(device_type_.c_str(), allocator), allocator);
  json.AddMember("model_type_",
                 rapidjson::Value(model_type_.c_str(), allocator), allocator);
  json.AddMember("is_path_", is_path_, allocator);
  rapidjson::Value mv(rapidjson::kArrayType);
  for (const auto& m : model_value_) {
    mv.PushBack(rapidjson::Value(m.c_str(), allocator), allocator);
  }
  json.AddMember("model_value_", mv, allocator);
  return base::kStatusCodeOk;
}

base::Status Sam3GraphParam::deserialize(rapidjson::Value& json) {
  if (json.HasMember("inference_type_") && json["inference_type_"].IsString()) {
    inference_type_ = json["inference_type_"].GetString();
  }
  if (json.HasMember("device_type_") && json["device_type_"].IsString()) {
    device_type_ = json["device_type_"].GetString();
  }
  if (json.HasMember("model_type_") && json["model_type_"].IsString()) {
    model_type_ = json["model_type_"].GetString();
  }
  if (json.HasMember("is_path_") && json["is_path_"].IsBool()) {
    is_path_ = json["is_path_"].GetBool();
  }
  if (json.HasMember("model_value_") && json["model_value_"].IsArray()) {
    model_value_.clear();
    const auto& arr = json["model_value_"];
    for (rapidjson::SizeType i = 0; i < arr.Size(); i++) {
      if (arr[i].IsString()) {
        model_value_.push_back(arr[i].GetString());
      }
    }
  }
  return base::kStatusCodeOk;
}

// ===================== SAM3Graph =====================

base::Status SAM3Graph::initDynamicsGraphNodes() {
  base::Status status = base::kStatusCodeOk;

  // Preprocessing
  preprocess_image_node_ =
      this->createNode<preprocess::CvtResizePadNormTrans>("preprocess_image");
  CHECK_IF_NULL_RETURN(preprocess_image_node_,
                       "Failed to create preprocess_image node");

  // Perception Encoder (Shared ViT backbone)
  perception_encoder_node_ = static_cast<Sam3PerceptionEncoder*>(
      this->createNode<Sam3PerceptionEncoder>("perception_encoder"));
  CHECK_IF_NULL_RETURN(perception_encoder_node_,
                       "Failed to create perception_encoder node");

  // Legacy concept encoder (backward compat)
  concept_encode_node_ = static_cast<Sam3ConceptEncodeNode*>(
      this->createNode<Sam3ConceptEncodeNode>("concept_encode"));
  CHECK_IF_NULL_RETURN(concept_encode_node_,
                       "Failed to create concept_encode node");

  // Language encoder (optional CLIP text encoder)
  language_encode_node_ = static_cast<Sam3LanguageEncodeNode*>(
      this->createNode<Sam3LanguageEncodeNode>("language_encode"));
  CHECK_IF_NULL_RETURN(language_encode_node_,
                       "Failed to create language_encode node");

  // Exemplar encoder
  exemplar_encode_node_ = static_cast<Sam3ExemplarEncodeNode*>(
      this->createNode<Sam3ExemplarEncodeNode>("exemplar_encode"));
  CHECK_IF_NULL_RETURN(exemplar_encode_node_,
                       "Failed to create exemplar_encode node");

  // NEW: DETR Detector Decoder
  detector_decoder_node_ = static_cast<Sam3DetectorDecoder*>(
      this->createNode<Sam3DetectorDecoder>("detector_decoder"));
  CHECK_IF_NULL_RETURN(detector_decoder_node_,
                       "Failed to create detector_decoder node");

  // NEW: Presence Head
  presence_head_node_ = static_cast<Sam3PresenceHead*>(
      this->createNode<Sam3PresenceHead>("presence_head"));
  CHECK_IF_NULL_RETURN(presence_head_node_,
                       "Failed to create presence_head node");

  // NEW: Concept Matcher
  concept_matcher_node_ = static_cast<Sam3ConceptMatcher*>(
      this->createNode<Sam3ConceptMatcher>("concept_matcher"));
  CHECK_IF_NULL_RETURN(concept_matcher_node_,
                       "Failed to create concept_matcher node");

  // NEW: Memory Encoder
  memory_encoder_node_ = static_cast<Sam3MemoryEncoder*>(
      this->createNode<Sam3MemoryEncoder>("memory_encoder"));
  CHECK_IF_NULL_RETURN(memory_encoder_node_,
                       "Failed to create memory_encoder node");

  // NEW: Tracker Mask Decoder
  tracker_decoder_node_ = static_cast<Sam3TrackerMaskDecoder*>(
      this->createNode<Sam3TrackerMaskDecoder>("tracker_decoder"));
  CHECK_IF_NULL_RETURN(tracker_decoder_node_,
                       "Failed to create tracker_decoder node");

  // NEW: Memory Manager
  memory_manager_node_ = static_cast<Sam3MemoryManager*>(
      this->createNode<Sam3MemoryManager>("memory_manager"));
  CHECK_IF_NULL_RETURN(memory_manager_node_,
                       "Failed to create memory_manager node");

  // Postprocess
  postprocess_node_ = static_cast<Sam3PostProcess*>(
      this->createNode<Sam3PostProcess>("postprocess"));
  CHECK_IF_NULL_RETURN(postprocess_node_, "Failed to create postprocess node");

  return status;
}

base::Status SAM3Graph::setInferParam(base::InferenceType inference_type,
                                      base::DeviceType device_type,
                                      base::ModelType model_type, bool is_path,
                                      std::vector<std::string>& model_value) {
  base::Status status = base::kStatusCodeOk;

  // model_value[0]: Perception Encoder (Shared ViT)
  if (model_value.size() > 0 && !model_value[0].empty()) {
    status = perception_encoder_node_->setInferParam(
        inference_type, device_type, model_type, is_path, model_value[0]);
    CHECK_IF_ERROR_RETURN(status, "Failed to set PE infer param");

    // Share PE with exemplar encoder
    exemplar_encode_node_->setSharedEncoder(
        perception_encoder_node_->getInferNode());
  }

  // model_value[1]: Detector Decoder (DETR)
  if (model_value.size() > 1 && !model_value[1].empty()) {
    status = detector_decoder_node_->setInferParam(
        inference_type, device_type, model_type, is_path, model_value[1]);
    CHECK_IF_ERROR_RETURN(status, "Failed to set detector infer param");
  }

  // model_value[2]: Memory Encoder
  if (model_value.size() > 2 && !model_value[2].empty()) {
    status = memory_encoder_node_->setInferParam(
        inference_type, device_type, model_type, is_path, model_value[2]);
    CHECK_IF_ERROR_RETURN(status, "Failed to set memory encoder param");
  }

  // model_value[3]: Tracker Mask Decoder
  if (model_value.size() > 3 && !model_value[3].empty()) {
    status = tracker_decoder_node_->setInferParam(
        inference_type, device_type, model_type, is_path, model_value[3]);
    CHECK_IF_ERROR_RETURN(status, "Failed to set tracker decoder param");
  }

  // model_value[4]: Text Encoder (optional)
  if (model_value.size() > 4 && !model_value[4].empty() &&
      language_encode_node_ != nullptr) {
    status = language_encode_node_->setTextEncoderInferParam(
        inference_type, device_type, model_type, is_path, model_value[4]);
    CHECK_IF_ERROR_RETURN(status, "Failed to set language encoder param");
  }

  return status;
}

base::Status SAM3Graph::defaultParam() {
  base::Status status = base::kStatusCodeOk;

  preprocess::CvtResizePadNormTransParam* preprocess_image_param =
      dynamic_cast<preprocess::CvtResizePadNormTransParam*>(
          preprocess_image_node_->getParam());
  CHECK_IF_NULL_RETURN(preprocess_image_param,
                       "Failed to get preprocess_image_param");
  preprocess_image_param->src_pixel_type_ = base::kPixelTypeBGR;
  preprocess_image_param->dst_pixel_type_ = base::kPixelTypeRGB;
  preprocess_image_param->interp_type_ = base::kInterpTypeLinear;
  preprocess_image_param->h_ = 1024;
  preprocess_image_param->w_ = 1024;
  preprocess_image_param->scale_[1] = 1.0f;
  preprocess_image_param->scale_[2] = 1.0f;
  preprocess_image_param->scale_[3] = 1.0f;
  preprocess_image_param->mean_[1] = 123.675;
  preprocess_image_param->mean_[2] = 116.28;
  preprocess_image_param->mean_[3] = 103.53;
  preprocess_image_param->std_[1] = 58.395;
  preprocess_image_param->std_[2] = 57.12;
  preprocess_image_param->std_[3] = 57.375;
  preprocess_image_param->normalize_ = false;
  preprocess_image_param->data_type_ = base::dataTypeOf<uint8_t>();

  Sam3PostParam* post_param =
      dynamic_cast<Sam3PostParam*>(postprocess_node_->getParam());
  if (post_param != nullptr) {
    post_param->score_threshold_ = 0.5f;
    post_param->presence_threshold_ = 0.3f;
    post_param->concept_threshold_ = 0.3f;
    post_param->model_h_ = 1024;
    post_param->model_w_ = 1024;
  }

  Sam3ConceptParam* concept_param =
      dynamic_cast<Sam3ConceptParam*>(concept_encode_node_->getParam());
  if (concept_param != nullptr) {
    concept_param->num_concepts_ = 1;
    concept_param->concept_dim_ = 512;
  }

  return status;
}

base::Status SAM3Graph::deserialize(rapidjson::Value& json) {
  // 1. Call base deserialize (reads param_ from JSON via Sam3GraphParam)
  base::Status status = dag::Graph::deserialize(json);
  if (status != base::kStatusCodeOk) {
    NNDEPLOY_LOGE("SAM3Graph base deserialize failed\n");
    return status;
  }

  // 2. Extract model paths from Sam3GraphParam and configure infer nodes
  Sam3GraphParam* graph_param = dynamic_cast<Sam3GraphParam*>(param_.get());
  if (graph_param == nullptr) {
    NNDEPLOY_LOGW(
        "SAM3Graph: param_ is not Sam3GraphParam, skipping model cfg");
    return base::kStatusCodeOk;
  }

  if (graph_param->model_value_.empty()) {
    NNDEPLOY_LOGW(
        "SAM3Graph: no model_value_ in param (JSON workflow may set "
        "models via C++ API)");
    return base::kStatusCodeOk;
  }

  // Convert string fields to enum types for setInferParam
  base::InferenceType inference_type =
      base::stringToInferenceType(graph_param->inference_type_);
  base::DeviceType device_type =
      base::stringToDeviceType(graph_param->device_type_);
  base::ModelType model_type =
      base::stringToModelType(graph_param->model_type_);

  NNDEPLOY_LOGI(
      "SAM3Graph::deserialize: configuring %zu model paths from JSON param",
      graph_param->model_value_.size());

  status =
      this->setInferParam(inference_type, device_type, model_type,
                          graph_param->is_path_, graph_param->model_value_);
  if (status != base::kStatusCodeOk) {
    NNDEPLOY_LOGE("SAM3Graph::deserialize: setInferParam failed\n");
    return status;
  }

  return base::kStatusCodeOk;
}

dag::Edge* SAM3Graph::fusePrompts(dag::Edge* text_emb,
                                  dag::Edge* exemplar_emb) {
  if (text_emb == nullptr && exemplar_emb == nullptr) {
    return nullptr;
  }
  if (text_emb != nullptr && exemplar_emb == nullptr) {
    return text_emb;  // text-only
  }
  if (text_emb == nullptr && exemplar_emb != nullptr) {
    return exemplar_emb;  // exemplar-only
  }

  // Hybrid: concatenate text + exemplar embeddings along the concept dimension
  device::Tensor* text_tensor = text_emb->getTensor(this);
  device::Tensor* exemplar_tensor = exemplar_emb->getTensor(this);
  if (text_tensor == nullptr || exemplar_tensor == nullptr) {
    return text_emb;  // fallback
  }

  device::TensorDesc text_desc = text_tensor->getDesc();
  device::TensorDesc exemplar_desc = exemplar_tensor->getDesc();
  int text_concepts = (text_desc.shape_.size() > 1) ? text_desc.shape_[1] : 0;
  int exemplar_concepts =
      (exemplar_desc.shape_.size() > 1) ? exemplar_desc.shape_[1] : 0;
  int total_concepts = text_concepts + exemplar_concepts;

  if (total_concepts <= 0) return text_emb;

  int dim = (text_desc.shape_.size() > 2) ? text_desc.shape_[2] : 256;

  device::Device* cur_device = device::getDefaultHostDevice();
  device::TensorDesc fused_desc;
  fused_desc.data_format_ = base::kDataFormatNCL;
  fused_desc.data_type_ = base::dataTypeOf<float>();
  fused_desc.shape_ = {1, total_concepts, dim};

  dag::Edge* fused_edge = this->createEdge("hybrid_prompt_fused");
  device::Tensor* fused_tensor = fused_edge->create(cur_device, fused_desc);
  if (fused_tensor == nullptr) return text_emb;

  float* fused_data = static_cast<float*>(fused_tensor->getData());
  float* text_data = static_cast<float*>(text_tensor->getData());
  float* exemplar_data = static_cast<float*>(exemplar_tensor->getData());

  memcpy(fused_data, text_data, text_concepts * dim * sizeof(float));
  memcpy(fused_data + text_concepts * dim, exemplar_data,
         exemplar_concepts * dim * sizeof(float));

  return fused_edge;
}

std::vector<dag::Edge*> SAM3Graph::forward(std::vector<dag::Edge*> inputs) {
  // Pipeline: Image → Preprocess → PE → DetectorDecoder → PresenceHead
  //           → ConceptMatcher → PostProcess
  // Prompt modes (auto-detected from inputs/params):
  //   [0]=image, [1]=text_concepts, [2]=exemplar_images (optional hybrid)

  // Step 1: Preprocess image
  std::vector<dag::Edge*> preproc_out = (*preprocess_image_node_)({inputs[0]});

  // Step 2: Perception Encoder (shared ViT backbone)
  std::vector<dag::Edge*> pe_out = (*perception_encoder_node_)(preproc_out);

  // Step 3: Encode concept prompts (text)
  dag::Edge* text_emb_edge = nullptr;
  if (inputs.size() > 1 && inputs[1] != nullptr) {
    std::vector<dag::Edge*> concept_emb = (*concept_encode_node_)({inputs[1]});
    text_emb_edge = concept_emb[0];

    if (language_encode_node_ != nullptr) {
      concept_emb = (*language_encode_node_)({inputs[1]});
      text_emb_edge = concept_emb[0];
    }
  }

  // Step 4: Encode exemplar prompts (images)
  dag::Edge* exemplar_emb_edge = nullptr;
  if ((inputs.size() > 2 && inputs[2] != nullptr) ||
      exemplar_encode_node_ != nullptr) {
    Sam3ExemplarParam* exemplar_param =
        dynamic_cast<Sam3ExemplarParam*>(exemplar_encode_node_->getParam());

    // Check if exemplars are in inputs[2] or via param
    bool has_exemplar_input = (inputs.size() > 2 && inputs[2] != nullptr);
    bool has_exemplar_param = (exemplar_param != nullptr &&
                               !exemplar_param->exemplar_images_.empty());

    if (has_exemplar_input || has_exemplar_param) {
      auto ex_inputs = has_exemplar_input ? std::vector<dag::Edge*>{inputs[2]}
                                          : std::vector<dag::Edge*>{};
      std::vector<dag::Edge*> exemplar_emb =
          (*exemplar_encode_node_)(ex_inputs);
      if (!exemplar_emb.empty() && exemplar_emb[0] != nullptr) {
        exemplar_emb_edge = exemplar_emb[0];
      }
    }
  }

  // Step 5: Hybrid prompt fusion (text + exemplar)
  dag::Edge* prompt_emb = fusePrompts(text_emb_edge, exemplar_emb_edge);
  if (prompt_emb == nullptr) {
    // Fallback: use empty concept embedding
    NNDEPLOY_LOGW("SAM 3: No prompt provided, using zero prompt");
    device::Device* cur_device = device::getDefaultHostDevice();
    device::TensorDesc fallback_desc;
    fallback_desc.data_format_ = base::kDataFormatNCL;
    fallback_desc.data_type_ = base::dataTypeOf<float>();
    fallback_desc.shape_ = {1, 1, 256};
    dag::Edge* fallback_edge = this->createEdge("fallback_prompt");
    device::Tensor* fallback_t =
        fallback_edge->create(cur_device, fallback_desc);
    memset(fallback_t->getData(), 0, fallback_t->getSize());
    prompt_emb = fallback_edge;
  }

  // Step 6: DETR Detector Decoder (PE features + fused prompt)
  std::vector<dag::Edge*> detector_out =
      (*detector_decoder_node_)({pe_out[0], prompt_emb});

  // Step 7: Presence Head (filter by binary presence score)
  std::vector<dag::Edge*> presence_out = (*presence_head_node_)(
      {detector_out[1], detector_out[0], detector_out[2]});

  // Step 8: Concept Matcher
  dag::Edge* concept_for_match = text_emb_edge;
  if (concept_for_match == nullptr && exemplar_emb_edge != nullptr) {
    concept_for_match = exemplar_emb_edge;
  }
  std::vector<dag::Edge*> match_out;
  if (presence_out.size() > 2 && presence_out[2] != nullptr &&
      concept_for_match != nullptr) {
    match_out = (*concept_matcher_node_)({presence_out[2], concept_for_match});
  }

  // Step 9: PostProcess
  dag::Edge* masks_for_post =
      (presence_out.size() > 1 && presence_out[1] != nullptr) ? presence_out[1]
                                                              : detector_out[0];
  dag::Edge* prescores_for_post =
      (presence_out.size() > 3 && presence_out[3] != nullptr) ? presence_out[3]
                                                              : detector_out[1];
  dag::Edge* match_for_post =
      (!match_out.empty() && match_out[0] != nullptr) ? match_out[0] : nullptr;

  std::vector<dag::Edge*> post_inputs = {masks_for_post};
  if (prescores_for_post != nullptr) post_inputs.push_back(prescores_for_post);
  if (match_for_post != nullptr) post_inputs.push_back(match_for_post);

  std::vector<dag::Edge*> postprocess_output =
      (*postprocess_node_)(post_inputs);

  return postprocess_output;
}

std::vector<dag::Edge*> SAM3Graph::forwardWithExemplars(
    cv::Mat& image, std::vector<cv::Mat>& exemplar_images) {
  // Set exemplar images via param
  Sam3ExemplarParam* exemplar_param =
      dynamic_cast<Sam3ExemplarParam*>(exemplar_encode_node_->getParam());
  if (exemplar_param != nullptr) {
    exemplar_param->exemplar_images_ = exemplar_images;
    exemplar_param->num_exemplars_ = static_cast<int>(exemplar_images.size());
  }

  dag::Edge* img_edge = this->createEdge("fewshot_img");
  img_edge->set(&image, false);
  return forward({img_edge, getInput(1), getInput(2)});
}

// ===================== Video Processing (memory-based tracking)
// =====================

base::Status SAM3Graph::processVideoFrame(cv::Mat& frame,
                                          device::Tensor* cached_concept_emb,
                                          int num_concepts,
                                          dag::Edge* output_edge,
                                          bool use_tracker) {
  base::Status status = base::kStatusCodeOk;

  dag::Edge* frame_input = this->createEdge("vf_input");
  dag::Edge* concept_emb_edge = this->createEdge("vf_conc_emb");

  frame_input->set(&frame, false);

  std::vector<dag::Edge*> preproc_outputs =
      (*preprocess_image_node_)({frame_input});

  std::vector<dag::Edge*> pe_outputs =
      (*perception_encoder_node_)(preproc_outputs);

  concept_emb_edge->set(cached_concept_emb, false);

  if (use_tracker && memory_manager_node_ != nullptr &&
      prev_frame_masks_ != nullptr) {
    // Memory-based tracker: PreviousMasks → MemoryEncoder → MemoryManager
    // → TrackerDecoder.  Avoids re-running the DETR detector on every frame.
    // Track IDs propagate from memory slot indices rather than IOU matching.
    dag::Edge* prev_masks_edge = this->createEdge("vf_prev_masks");
    prev_masks_edge->set(prev_frame_masks_, false);

    std::vector<dag::Edge*> mem_outputs =
        (*memory_encoder_node_)({pe_outputs[0], prev_masks_edge});

    std::vector<dag::Edge*> mem_bank =
        (*memory_manager_node_)({mem_outputs[0]});

    std::vector<dag::Edge*> tracker_outputs =
        (*tracker_decoder_node_)({pe_outputs[0], mem_bank[0]});

    dag::Edge* empty_pres = this->createEdge("vf_pres_empty");
    dag::Edge* empty_concept = this->createEdge("vf_concept_empty");
    std::vector<dag::Edge*> postproc_outputs =
        (*postprocess_node_)({tracker_outputs[0], empty_pres, empty_concept});

    cv::Mat* result = postproc_outputs[0]->get<cv::Mat>(this);
    if (result != nullptr) {
      prev_frame_masks_ = new cv::Mat(result->clone());
      output_edge->set(result, false);
    }
  } else {
    // Detector path: PE → DETR → PresenceHead → PostProcess
    std::vector<dag::Edge*> detector_outputs =
        (*detector_decoder_node_)({pe_outputs[0], concept_emb_edge});

    std::vector<dag::Edge*> presence_outputs = (*presence_head_node_)(
        {detector_outputs[1], detector_outputs[0], detector_outputs[2]});

    dag::Edge* masks_for_post =
        (presence_outputs.size() > 1 && presence_outputs[1] != nullptr)
            ? presence_outputs[1]
            : detector_outputs[0];
    dag::Edge* prescores_for_post =
        (presence_outputs.size() > 3 && presence_outputs[3] != nullptr)
            ? presence_outputs[3]
            : detector_outputs[1];

    std::vector<dag::Edge*> postproc_inputs = {masks_for_post};
    if (prescores_for_post != nullptr)
      postproc_inputs.push_back(prescores_for_post);

    std::vector<dag::Edge*> postproc_outputs =
        (*postprocess_node_)(postproc_inputs);

    cv::Mat* result = postproc_outputs[0]->get<cv::Mat>(this);
    if (result != nullptr) {
      if (prev_frame_masks_ != nullptr) {
        delete prev_frame_masks_;
      }
      prev_frame_masks_ = new cv::Mat(result->clone());
      output_edge->set(result, false);
    }
  }

  return base::kStatusCodeOk;
}

std::vector<std::vector<dag::Edge*>> SAM3Graph::forwardVideoWithTextPrompt(
    std::vector<cv::Mat>& video_frames, std::vector<std::string>& concepts,
    std::vector<std::vector<float>> prompts) {
  std::vector<std::vector<dag::Edge*>> results;
  int num_concepts = static_cast<int>(concepts.size());

  if (num_concepts == 0) {
    NNDEPLOY_LOGE("No concepts provided for video text prompting.");
    return results;
  }

  // Set concepts
  Sam3ConceptParam* concept_param =
      dynamic_cast<Sam3ConceptParam*>(concept_encode_node_->getParam());
  if (concept_param != nullptr) {
    concept_param->concepts_ = concepts;
    concept_param->num_concepts_ = num_concepts;
    concept_param->concept_dim_ = 512;
  }

  // Cache concept embeddings once
  if (!concept_embeddings_cached_valid_) {
    dag::Edge* concept_input = this->createEdge("video_concept_input");
    dag::Edge* concept_output = this->createEdge("video_concept_output");

    concept_input->set(concept_param, false);

    std::vector<dag::Edge*> concept_emb;
    if (language_encode_node_ != nullptr) {
      concept_emb = (*language_encode_node_)({concept_input});
    } else {
      concept_emb = (*concept_encode_node_)({concept_input});
    }

    device::Tensor* emb_tensor = concept_emb[0]->getTensor(this);
    if (emb_tensor != nullptr) {
      concept_embeddings_cached_ =
          std::make_shared<device::Tensor>(*emb_tensor);
      concept_embeddings_cached_valid_ = true;
      NNDEPLOY_LOGI("SAM 3 Video: Cached %d concept embeddings (dim=%d)",
                    num_concepts, concept_param->concept_dim_);
    } else {
      NNDEPLOY_LOGE("SAM 3 Video: Failed to get concept embeddings");
      return results;
    }
  }

  if (concept_embeddings_cached_ == nullptr) {
    return results;
  }

  // Process each frame
  for (size_t i = 0; i < video_frames.size(); ++i) {
    dag::Edge* frame_output =
        this->createEdge("video_frame_out_" + std::to_string(i));
    dag::Edge* track_out =
        this->createEdge("video_track_out_" + std::to_string(i));

    // First frame: use detector. Subsequent frames: use tracker
    bool use_tracker = (i > 0) && (memory_manager_node_ != nullptr);

    base::Status status =
        processVideoFrame(video_frames[i], concept_embeddings_cached_.get(),
                          num_concepts, frame_output, use_tracker);

    if (status != base::kStatusCodeOk) {
      NNDEPLOY_LOGE("SAM 3 Video: Frame %zu processing failed", i);
      continue;
    }

    cv::Mat* result_mask = frame_output->get<cv::Mat>(this);
    if (result_mask != nullptr) {
      bool is_tracker_frame = (i > 0 && memory_manager_node_ != nullptr);
      active_instances_.clear();
      cv::Mat labels, stats, centroids;
      int num_components = cv::connectedComponentsWithStats(
          *result_mask, labels, stats, centroids);

      for (int c = 1; c < num_components; ++c) {
        DetectionInstance inst;
        inst.track_id = is_tracker_frame ? (c - 1) : assignTrackId();
        inst.bbox.x = stats.at<int>(c, cv::CC_STAT_LEFT);
        inst.bbox.y = stats.at<int>(c, cv::CC_STAT_TOP);
        inst.bbox.width = stats.at<int>(c, cv::CC_STAT_WIDTH);
        inst.bbox.height = stats.at<int>(c, cv::CC_STAT_HEIGHT);
        inst.concept_idx = 0;
        inst.final_score = 1.0f;
        inst.age = is_tracker_frame ? (active_frame_count_ + 1) : 1;
        inst.mask = (*result_mask)(inst.bbox).clone();
        active_instances_.push_back(inst);
      }
    }

    std::vector<dag::Edge*> frame_result = {frame_output, track_out};
    results.push_back(frame_result);
    active_frame_count_++;

    NNDEPLOY_LOGI("SAM 3 Video: Frame %zu/%zu processed, %zu active instances",
                  i + 1, video_frames.size(), active_instances_.size());
  }

  return results;
}

std::vector<std::vector<dag::Edge*>> SAM3Graph::forwardVideoWithExemplars(
    std::vector<cv::Mat>& video_frames, std::vector<cv::Mat>& exemplar_images) {
  std::vector<std::string> concept_names;
  for (size_t i = 0; i < exemplar_images.size(); ++i) {
    concept_names.push_back("exemplar_" + std::to_string(i));
  }
  return forwardVideoWithTextPrompt(video_frames, concept_names);
}

// Node registration
REGISTER_NODE("nndeploy::segment::Sam3LanguageEncodeNode",
              Sam3LanguageEncodeNode);
REGISTER_NODE("nndeploy::segment::Sam3ConceptEncodeNode",
              Sam3ConceptEncodeNode);
REGISTER_NODE("nndeploy::segment::Sam3ExemplarEncodeNode",
              Sam3ExemplarEncodeNode);
REGISTER_NODE("nndeploy::segment::Sam3PostProcess", Sam3PostProcess);
REGISTER_NODE("nndeploy::segment::Sam3PerceptionEncoder",
              Sam3PerceptionEncoder);
REGISTER_NODE("nndeploy::segment::Sam3DetectorDecoder", Sam3DetectorDecoder);
REGISTER_NODE("nndeploy::segment::Sam3PresenceHead", Sam3PresenceHead);
REGISTER_NODE("nndeploy::segment::Sam3ConceptMatcher", Sam3ConceptMatcher);
REGISTER_NODE("nndeploy::segment::Sam3MemoryEncoder", Sam3MemoryEncoder);
REGISTER_NODE("nndeploy::segment::Sam3TrackerMaskDecoder",
              Sam3TrackerMaskDecoder);
REGISTER_NODE("nndeploy::segment::Sam3MemoryManager", Sam3MemoryManager);
REGISTER_NODE("nndeploy::segment::Sam3SimpleImageEncoder",
              Sam3SimpleImageEncoder);
REGISTER_NODE("nndeploy::segment::Sam3SimpleLanguageEncoder",
              Sam3SimpleLanguageEncoder);
REGISTER_NODE("nndeploy::segment::Sam3SimpleDecoder", Sam3SimpleDecoder);
REGISTER_NODE("nndeploy::segment::Sam3SimplePostprocess",
              Sam3SimplePostprocess);
REGISTER_NODE("nndeploy::segment::Sam3SimpleGraph", Sam3SimpleGraph);
REGISTER_NODE("nndeploy::segment::SAM3Graph", SAM3Graph);

}  // namespace segment
}  // namespace nndeploy
