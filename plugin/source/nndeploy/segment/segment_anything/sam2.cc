#include "nndeploy/segment/segment_anything/sam2.h"

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

// ===================== SAM2PointsParam =====================

base::Status SAM2PointsParam::serialize(
    rapidjson::Value& json, rapidjson::Document::AllocatorType& allocator) {
  json.SetObject();
  rapidjson::Value points_array(rapidjson::kArrayType);
  for (size_t i = 0; i < points_.size(); ++i) {
    points_array.PushBack(points_[i], allocator);
  }
  json.AddMember("points", points_array, allocator);

  rapidjson::Value labels_array(rapidjson::kArrayType);
  for (size_t i = 0; i < labels_.size(); ++i) {
    labels_array.PushBack(labels_[i], allocator);
  }
  json.AddMember("labels", labels_array, allocator);

  json.AddMember("ori_width_", ori_width_, allocator);
  json.AddMember("ori_height_", ori_height_, allocator);
  json.AddMember("version_", version_, allocator);
  return base::kStatusCodeOk;
}

base::Status SAM2PointsParam::deserialize(rapidjson::Value& json) {
  if (json.HasMember("points") && json["points"].IsArray()) {
    points_.clear();
    const auto& arr = json["points"];
    for (rapidjson::SizeType i = 0; i < arr.Size(); i++) {
      if (arr[i].IsFloat()) {
        points_.push_back(arr[i].GetFloat());
      } else if (arr[i].IsInt()) {
        points_.push_back(static_cast<float>(arr[i].GetInt()));
      }
    }
  }
  if (json.HasMember("labels") && json["labels"].IsArray()) {
    labels_.clear();
    const auto& arr = json["labels"];
    for (rapidjson::SizeType i = 0; i < arr.Size(); i++) {
      if (arr[i].IsFloat()) {
        labels_.push_back(arr[i].GetFloat());
      } else if (arr[i].IsInt()) {
        labels_.push_back(static_cast<float>(arr[i].GetInt()));
      }
    }
  }
  if (json.HasMember("ori_width_") && json["ori_width_"].IsInt()) {
    ori_width_ = json["ori_width_"].GetInt();
  } else if (json.HasMember("ori_width") && json["ori_width"].IsInt()) {
    ori_width_ = json["ori_width"].GetInt();
  }
  if (json.HasMember("ori_height_") && json["ori_height_"].IsInt()) {
    ori_height_ = json["ori_height_"].GetInt();
  } else if (json.HasMember("ori_height") && json["ori_height"].IsInt()) {
    ori_height_ = json["ori_height"].GetInt();
  }
  if (json.HasMember("version_") && json["version_"].IsInt()) {
    version_ = json["version_"].GetInt();
  }
  return base::kStatusCodeOk;
}

// ===================== SAM2PostParam =====================

base::Status SAM2PostParam::serialize(
    rapidjson::Value& json, rapidjson::Document::AllocatorType& allocator) {
  json.SetObject();
  json.AddMember("score_threshold_", score_threshold_, allocator);
  json.AddMember("model_h_", model_h_, allocator);
  json.AddMember("model_w_", model_w_, allocator);
  return base::kStatusCodeOk;
}

base::Status SAM2PostParam::deserialize(rapidjson::Value& json) {
  if (json.HasMember("score_threshold_") &&
      json["score_threshold_"].IsFloat()) {
    score_threshold_ = json["score_threshold_"].GetFloat();
  }
  if (json.HasMember("model_h_") && json["model_h_"].IsInt()) {
    model_h_ = json["model_h_"].GetInt();
  }
  if (json.HasMember("model_w_") && json["model_w_"].IsInt()) {
    model_w_ = json["model_w_"].GetInt();
  }
  return base::kStatusCodeOk;
}

// ===================== SAM2PointNode =====================

base::Status SAM2PointNode::run() {
  SAM2PointsParam* param = dynamic_cast<SAM2PointsParam*>(param_.get());
  CHECK_IF_NULL_RETURN(param, "SAM2PointsParam is null");

  // Prefer input param if available
  if (!inputs_.empty() && inputs_[0] != nullptr) {
    SAM2PointsParam* input_param = inputs_[0]->get<SAM2PointsParam>(this);
    if (input_param != nullptr) {
      param = input_param;
    }
  }

  int points_num = static_cast<int>(param->points_.size()) / 2;
  if (points_num <= 0 || param->labels_.empty()) {
    NNDEPLOY_LOGE("Invalid points: %d pairs, %zu labels", points_num,
                  param->labels_.size());
    return base::kStatusCodeErrorInvalidValue;
  }
  if (points_num != static_cast<int>(param->labels_.size())) {
    NNDEPLOY_LOGE("Points/labels mismatch: %d vs %zu", points_num,
                  param->labels_.size());
    return base::kStatusCodeErrorInvalidValue;
  }

  device::Device* cur_device = device::getDefaultHostDevice();

  // Scale points to 1024x1024 model space
  // Guard: protect against division by zero when ori_height_/ori_width_ are 0
  const int model_size = 1024;
  int origin_h = param->ori_height_;
  int origin_w = param->ori_width_;
  float scale_width = 1.0f;
  float scale_height = 1.0f;
  if (origin_h > 0 && origin_w > 0) {
    float scale_h = static_cast<float>(model_size) / origin_h;
    float scale_w = static_cast<float>(model_size) / origin_w;
    int new_h = model_size, new_w = model_size;
    if (scale_h < scale_w) {
      new_w = static_cast<int>(std::round(origin_w * scale_h));
    } else {
      new_h = static_cast<int>(std::round(origin_h * scale_w));
    }
    scale_width = static_cast<float>(new_w) / origin_w;
    scale_height = static_cast<float>(new_h) / origin_h;
  }

  // point_coords tensor (1, N, 2)
  device::TensorDesc coords_desc;
  coords_desc.data_format_ = base::kDataFormatNCL;
  coords_desc.data_type_ = base::dataTypeOf<float>();
  coords_desc.shape_ = {1, points_num, 2};
  device::Tensor* point_coords = outputs_[0]->create(cur_device, coords_desc);
  CHECK_IF_NULL_RETURN(point_coords, "Failed to create coords tensor");
  float* coords_data = static_cast<float*>(point_coords->getData());
  if (coords_data != nullptr) {
    for (int i = 0; i < points_num; ++i) {
      coords_data[2 * i + 0] = param->points_[2 * i + 0] * scale_width;
      coords_data[2 * i + 1] = param->points_[2 * i + 1] * scale_height;
    }
  }

  // point_labels tensor (1, N)
  device::TensorDesc labels_desc;
  labels_desc.data_format_ = base::kDataFormatNC;
  labels_desc.data_type_ = base::dataTypeOf<float>();
  labels_desc.shape_ = {1, points_num};
  device::Tensor* point_labels = outputs_[1]->create(cur_device, labels_desc);
  CHECK_IF_NULL_RETURN(point_labels, "Failed to create labels tensor");
  float* labels_data = static_cast<float*>(point_labels->getData());
  if (labels_data != nullptr) {
    for (int i = 0; i < points_num; ++i) {
      labels_data[i] = param->labels_[i];
    }
  }

  // orig_im_size tensor (2,)
  device::TensorDesc size_desc;
  size_desc.data_format_ = base::kDataFormatN;
  size_desc.data_type_ = base::dataTypeOf<float>();
  size_desc.shape_ = {2};
  device::Tensor* orig_im_size = outputs_[2]->create(cur_device, size_desc);
  CHECK_IF_NULL_RETURN(orig_im_size, "Failed to create size tensor");
  float* size_data = static_cast<float*>(orig_im_size->getData());
  if (size_data != nullptr) {
    size_data[0] = static_cast<float>(origin_h);
    size_data[1] = static_cast<float>(origin_w);
  }

  return base::kStatusCodeOk;
}

// ===================== SAM2MaskNode =====================

base::Status SAM2MaskNode::run() {
  device::Device* cur_device = device::getDefaultHostDevice();

  // mask_input (1,1,256,256) - all zeros (no mask prompt)
  device::TensorDesc mask_desc;
  mask_desc.data_format_ = base::kDataFormatNCHW;
  mask_desc.data_type_ = base::dataTypeOf<float>();
  mask_desc.shape_ = {1, 1, 256, 256};
  device::Tensor* mask_input = outputs_[0]->create(cur_device, mask_desc);
  CHECK_IF_NULL_RETURN(mask_input, "Failed to create mask tensor");
  if (mask_input->getData() != nullptr) {
    memset(mask_input->getData(), 0, mask_input->getSize());
  }

  // has_mask_input (1,) - all zeros (no mask)
  device::TensorDesc has_mask_desc;
  has_mask_desc.data_format_ = base::kDataFormatN;
  has_mask_desc.data_type_ = base::dataTypeOf<float>();
  has_mask_desc.shape_ = {1};
  device::Tensor* has_mask = outputs_[1]->create(cur_device, has_mask_desc);
  CHECK_IF_NULL_RETURN(has_mask, "Failed to create has_mask tensor");
  if (has_mask->getData() != nullptr) {
    memset(has_mask->getData(), 0, has_mask->getSize());
  }

  return base::kStatusCodeOk;
}

// ===================== SAM2PostProcess =====================

base::Status SAM2PostProcess::defaultParam() {
  SAM2PostParam* param = dynamic_cast<SAM2PostParam*>(param_.get());
  param->score_threshold_ = 0.0f;
  param->model_h_ = 1024;
  param->model_w_ = 1024;
  return base::kStatusCodeOk;
}

base::Status SAM2PostProcess::run() {
  SAM2PostParam* param = dynamic_cast<SAM2PostParam*>(param_.get());

  device::Tensor* masks_tensor = inputs_[0]->getTensor(this);
  device::Tensor* iou_tensor = inputs_[1]->getTensor(this);
  CHECK_IF_NULL_RETURN(masks_tensor, "Masks tensor is null");
  CHECK_IF_NULL_RETURN(iou_tensor, "IOU tensor is null");

  device::TensorDesc masks_desc = masks_tensor->getDesc();
  int result_num = masks_desc.shape_[1];
  int height = masks_desc.shape_[2];
  int width = masks_desc.shape_[3];

  float* masks_data = static_cast<float*>(masks_tensor->getData());
  float* iou_data = static_cast<float*>(iou_tensor->getData());
  int offset = height * width;

  if (result_num <= 0) {
    cv::Mat* blank = new cv::Mat(height, width, CV_8UC1, cv::Scalar(0));
    outputs_[0]->set(blank, false);
    return base::kStatusCodeOk;
  }

  int best_idx = 0;
  float best_iou = 0.0f;
  for (int i = 0; i < result_num; ++i) {
    if (iou_data[i] > best_iou) {
      best_iou = iou_data[i];
      best_idx = i;
    }
  }

  if (best_iou < param->score_threshold_) {
    cv::Mat* blank = new cv::Mat(height, width, CV_8UC1, cv::Scalar(0));
    outputs_[0]->set(blank, false);
    return base::kStatusCodeOk;
  }

  cv::Mat* result = new cv::Mat();
  cv::Mat mask(height, width, CV_32FC1, masks_data + best_idx * offset);
  mask.convertTo(*result, CV_8UC1, 255.0f);
  outputs_[0]->set(result, false);

  return base::kStatusCodeOk;
}

// ===================== SAM2MemoryNode =====================

// defaultParam() and run() are defined inline in sam2.h

void SAM2MemoryNode::storeMask(device::Tensor* masks_tensor) {
  if (masks_tensor == nullptr) {
    NNDEPLOY_LOGE("SAM2MemoryNode: null masks tensor");
    return;
  }
  device::TensorDesc desc = masks_tensor->getDesc();
  if (desc.shape_.size() < 4) {
    NNDEPLOY_LOGE("SAM2MemoryNode: expected 4D mask tensor, got %zuD",
                  desc.shape_.size());
    return;
  }
  // ONNX decoder 输出 masks 形状: (1, N, 256, 256)
  // 取第一个 mask 通道 [0, 0, :, :]
  prev_mask_h_ = desc.shape_[2];
  prev_mask_w_ = desc.shape_[3];
  size_t channel_size =
      static_cast<size_t>(prev_mask_h_) * static_cast<size_t>(prev_mask_w_);
  prev_mask_data_.resize(channel_size);

  float* data = static_cast<float*>(masks_tensor->getData());
  if (data == nullptr) {
    NNDEPLOY_LOGE("SAM2MemoryNode: null mask data");
    return;
  }
  memcpy(prev_mask_data_.data(), data, channel_size * sizeof(float));

  frame_count_++;
  NNDEPLOY_LOGI("SAM 2 Memory: stored mask from frame %d (%dx%d)", frame_count_,
                prev_mask_h_, prev_mask_w_);
}

void SAM2MemoryNode::fillMaskEdge(dag::Edge* mask_edge,
                                  dag::Edge* has_mask_edge) {
  if (mask_edge == nullptr || has_mask_edge == nullptr) {
    NNDEPLOY_LOGE("SAM2MemoryNode: null edge pointer");
    return;
  }
  if (prev_mask_data_.empty()) {
    NNDEPLOY_LOGE("SAM2MemoryNode: no stored mask to fill");
    return;
  }

  device::Device* cur_device = device::getDefaultHostDevice();

  // Fill mask_input edge with stored mask
  device::TensorDesc mask_desc;
  mask_desc.data_format_ = base::kDataFormatNCHW;
  mask_desc.data_type_ = base::dataTypeOf<float>();
  mask_desc.shape_ = {1, 1, prev_mask_h_, prev_mask_w_};
  device::Tensor* mask_tensor = mask_edge->create(cur_device, mask_desc);
  if (mask_tensor != nullptr) {
    memcpy(mask_tensor->getData(), prev_mask_data_.data(),
           prev_mask_data_.size() * sizeof(float));
  }

  // Fill has_mask_input edge with 1.0 (indicating a valid mask is provided)
  device::TensorDesc flag_desc;
  flag_desc.data_format_ = base::kDataFormatN;
  flag_desc.data_type_ = base::dataTypeOf<float>();
  flag_desc.shape_ = {1};
  device::Tensor* flag_tensor = has_mask_edge->create(cur_device, flag_desc);
  if (flag_tensor != nullptr) {
    float* flag_data = static_cast<float*>(flag_tensor->getData());
    flag_data[0] = 1.0f;
  }

  NNDEPLOY_LOGI("SAM 2 Memory: injected stored mask into decoder input");
}

// ===================== SAM2Graph =====================

base::Status SAM2Graph::setInferParam(base::InferenceType inference_type,
                                      base::DeviceType device_type,
                                      base::ModelType model_type, bool is_path,
                                      std::vector<std::string>& model_value) {
  base::Status status = base::kStatusCodeOk;

  encoder_infer_param_.inference_type_ = inference_type;
  encoder_infer_param_.device_type_ = device_type;
  encoder_infer_param_.model_type_ = model_type;
  encoder_infer_param_.is_path_ = is_path;
  encoder_infer_param_.model_value_ = {model_value[0]};
  status = encoder_infer_node_->setInferenceType(
      encoder_infer_param_.inference_type_);
  CHECK_IF_ERROR_RETURN(status,
                        "Failed to set inference type for encoder_infer");
  status = encoder_infer_node_->setParam(&encoder_infer_param_);
  CHECK_IF_ERROR_RETURN(status, "Failed to set param for encoder_infer node");

  decoder_infer_param_.inference_type_ = inference_type;
  decoder_infer_param_.device_type_ = device_type;
  decoder_infer_param_.model_type_ = model_type;
  decoder_infer_param_.is_path_ = is_path;
  decoder_infer_param_.model_value_ = {model_value[1]};
  decoder_infer_param_.is_dynamic_shape_ = true;
  decoder_infer_param_.max_shape_.insert({"image_embed", {1, 256, 64, 64}});
  decoder_infer_param_.max_shape_.insert(
      {"high_res_feats_0", {1, 32, 256, 256}});
  decoder_infer_param_.max_shape_.insert(
      {"high_res_feats_1", {1, 64, 128, 128}});
  decoder_infer_param_.max_shape_.insert({"point_coords", {1, 3, 2}});
  decoder_infer_param_.max_shape_.insert({"point_labels", {1, 3}});
  status = decoder_infer_node_->setInferenceType(
      decoder_infer_param_.inference_type_);
  CHECK_IF_ERROR_RETURN(status,
                        "Failed to set inference type for decoder_infer");
  status = decoder_infer_node_->setParam(&decoder_infer_param_);
  CHECK_IF_ERROR_RETURN(status, "Failed to set param for decoder_infer node");

  return status;
}

base::Status SAM2Graph::defaultParam() {
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
  preprocess_image_param->scale_[0] = 1.0f;
  preprocess_image_param->scale_[1] = 1.0f;
  preprocess_image_param->scale_[2] = 1.0f;
  preprocess_image_param->scale_[3] = 1.0f;
  preprocess_image_param->mean_[0] = 123.675;
  preprocess_image_param->mean_[1] = 116.28;
  preprocess_image_param->mean_[2] = 103.53;
  preprocess_image_param->std_[0] = 58.395;
  preprocess_image_param->std_[1] = 57.12;
  preprocess_image_param->std_[2] = 57.375;
  preprocess_image_param->normalize_ = true;
  preprocess_image_param->data_type_ = base::dataTypeOf<float>();

  return status;
}

base::Status SAM2Graph::initDynamicsGraphNodes() {
  base::Status status = base::kStatusCodeOk;

  preprocess_image_node_ =
      this->createNode<preprocess::CvtResizePadNormTrans>("preprocess_image");
  CHECK_IF_NULL_RETURN(preprocess_image_node_,
                       "Failed to create preprocess_image node");

  encoder_infer_node_ = dynamic_cast<infer::Infer*>(
      this->createNode<infer::Infer>("encoder_infer"));
  CHECK_IF_NULL_RETURN(encoder_infer_node_,
                       "Failed to create encoder_infer node");

  preprocess_point_node_ = this->createNode<SAM2PointNode>("preprocess_point");
  CHECK_IF_NULL_RETURN(preprocess_point_node_,
                       "Failed to create preprocess_point node");

  preprocess_mask_node_ = this->createNode<SAM2MaskNode>("preprocess_mask");
  CHECK_IF_NULL_RETURN(preprocess_mask_node_,
                       "Failed to create preprocess_mask node");

  decoder_infer_node_ = dynamic_cast<infer::Infer*>(
      this->createNode<infer::Infer>("decoder_infer"));
  CHECK_IF_NULL_RETURN(decoder_infer_node_,
                       "Failed to create decoder_infer node");

  // SAM 2 decoder input names (multi-scale features + prompts)
  decoder_infer_node_->setInputName("image_embed", 0);
  decoder_infer_node_->setInputName("high_res_feats_0", 1);
  decoder_infer_node_->setInputName("high_res_feats_1", 2);
  decoder_infer_node_->setInputName("point_coords", 3);
  decoder_infer_node_->setInputName("point_labels", 4);
  decoder_infer_node_->setInputName("mask_input", 5);
  decoder_infer_node_->setInputName("has_mask_input", 6);
  // 注意：实际 ONNX 解码器只有 7 个输入（无 orig_im_size）和 2 个输出（无
  // low_res_masks）
  decoder_infer_node_->setOutputName("masks", 0);
  decoder_infer_node_->setOutputName("iou_predictions", 1);

  postprocess_node_ = this->createNode<SAM2PostProcess>("postprocess");
  CHECK_IF_NULL_RETURN(postprocess_node_, "Failed to create postprocess node");

  // Optional memory node for video tracking
  memory_node_ =
      dynamic_cast<SAM2MemoryNode*>(this->createNode<SAM2MemoryNode>("memory"));
  CHECK_IF_NULL_RETURN(memory_node_, "Failed to create memory node");

  // Video tracking dynamic edges for mask propagation
  video_mask_input_edge_ = this->createEdge("video_mask_input");
  video_has_mask_edge_ = this->createEdge("video_has_mask");

  return status;
}

std::vector<dag::Edge*> SAM2Graph::forwardVideo(std::vector<dag::Edge*> inputs,
                                                bool reset_memory) {
  if (reset_memory) {
    memory_node_->reset();
  }

  // 1. Image encoder
  std::vector<dag::Edge*> encoder_input =
      (*preprocess_image_node_)({inputs[0]});
  std::vector<dag::Edge*> encoder_outputs =
      (*encoder_infer_node_)(encoder_input);

  // 2. Point prompt preprocessing
  std::vector<dag::Edge*> point_result = (*preprocess_point_node_)({inputs[1]});

  // 3. Mask input: use stored mask from memory if available, else zero mask
  std::vector<dag::Edge*> mask_result;
  if (memory_node_->hasPrevMask()) {
    memory_node_->fillMaskEdge(video_mask_input_edge_, video_has_mask_edge_);
    mask_result = {video_mask_input_edge_, video_has_mask_edge_};
  } else {
    mask_result = (*preprocess_mask_node_)();
  }

  // 4. Decoder with 7 inputs (ONNX format)
  std::vector<dag::Edge*> decoder_input = {
      encoder_outputs[2],  // image_embed (ONNX 输出索引 2)
      encoder_outputs[0],  // high_res_feats_0 (ONNX 输出索引 0)
      encoder_outputs[1],  // high_res_feats_1 (ONNX 输出索引 1)
      point_result[0],     // point_coords
      point_result[1],     // point_labels
      mask_result[0],      // mask_input (stored mask or zero)
      mask_result[1],      // has_mask_input (1.0 or 0.0)
  };
  std::vector<dag::Edge*> decoder_output =
      (*decoder_infer_node_)(decoder_input);

  // 5. Store decoder output masks in memory for next frame's mask propagation
  //    decoder_output[0] is the "masks" output tensor (1, N, 256, 256)
  device::Tensor* masks_tensor =
      decoder_infer_node_->getOutput(0)->getTensor(decoder_infer_node_);
  if (masks_tensor != nullptr) {
    memory_node_->storeMask(masks_tensor);
  }

  // 6. Postprocess: select best mask by IoU score
  std::vector<dag::Edge*> postprocess_output =
      (*postprocess_node_)(decoder_output);
  return postprocess_output;
}

// Node registration
REGISTER_NODE("nndeploy::segment::SAM2PointNode", SAM2PointNode);
REGISTER_NODE("nndeploy::segment::SAM2MaskNode", SAM2MaskNode);
REGISTER_NODE("nndeploy::segment::SAM2PostProcess", SAM2PostProcess);
REGISTER_NODE("nndeploy::segment::SAM2MemoryNode", SAM2MemoryNode);
REGISTER_NODE("nndeploy::segment::SAM2Graph", SAM2Graph);

}  // namespace segment
}  // namespace nndeploy
