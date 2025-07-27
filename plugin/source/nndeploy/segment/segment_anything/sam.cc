#include "nndeploy/segment/segment_anything/sam.h"

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

base::Status SAMPointsParam::serialize(
    rapidjson::Value &json, rapidjson::Document::AllocatorType &allocator) {
  json.SetObject();
  rapidjson::Value points_array(rapidjson::kArrayType);
  for (const auto &point : points_) {
    points_array.PushBack(point, allocator);
  }
  json.AddMember("points", points_array, allocator);

  rapidjson::Value labels_array(rapidjson::kArrayType);
  for (const auto &label : labels_) {
    labels_array.PushBack(label, allocator);
  }
  json.AddMember("labels", labels_array, allocator);

  json.AddMember("ori_width", ori_width, allocator);
  json.AddMember("ori_height", ori_height, allocator);
  json.AddMember("version", version_, allocator);

  return base::kStatusCodeOk;
}

base::Status SAMPointsParam::deserialize(rapidjson::Value &json) {
  // 反序列化 points_ 数组
  if (json.HasMember("points") && json["points"].IsArray()) {
    points_.clear();
    const rapidjson::Value &points_array = json["points"];
    for (rapidjson::SizeType i = 0; i < points_array.Size(); i++) {
      if (points_array[i].IsFloat()) {
        points_.push_back(points_array[i].GetFloat());
      }
    }
  }

  // 反序列化 labels_ 数组
  if (json.HasMember("labels") && json["labels"].IsArray()) {
    labels_.clear();
    const rapidjson::Value &labels_array = json["labels"];
    for (rapidjson::SizeType i = 0; i < labels_array.Size(); i++) {
      if (labels_array[i].IsFloat()) {
        labels_.push_back(labels_array[i].GetFloat());
      }
    }
  }

  // 反序列化其他成员
  if (json.HasMember("ori_width") && json["ori_width"].IsInt()) {
    ori_width = json["ori_width"].GetInt();
  }

  if (json.HasMember("ori_height") && json["ori_height"].IsInt()) {
    ori_height = json["ori_height"].GetInt();
  }

  if (json.HasMember("version") && json["version"].IsInt()) {
    version_ = json["version"].GetInt();
  }

  return base::kStatusCodeOk;
}

class SAMPointNode : public dag::Node {
 public:
  SAMPointNode(const std::string &name) : dag::Node(name) {
    this->setInputTypeInfo<SAMPointsParam>();
    this->setOutputTypeInfo<device::Tensor>();
    this->setOutputTypeInfo<device::Tensor>();
    this->setOutputTypeInfo<device::Tensor>();
  }
  SAMPointNode(const std::string &name, std::vector<dag::Edge *> inputs,
               std::vector<dag::Edge *> outputs)
      : dag::Node(name, inputs, outputs) {
    key_ = "nndeploy::segment::SAMPointNode";
    desc_ = "Segment Anything Point Node for image segmentation tasks.";
    this->setInputTypeInfo<SAMPointsParam>();
    this->setOutputTypeInfo<device::Tensor>();
    this->setOutputTypeInfo<device::Tensor>();
    this->setOutputTypeInfo<device::Tensor>();
  }
  virtual ~SAMPointNode() {}

  base::Status run() override {
    SAMPointsParam *param =
        (SAMPointsParam *)inputs_[0]->get<SAMPointsParam>(this);
    CHECK_IF_NULL_RETURN(param, "Failed to get SAMPointsParam from input");

    base::Status status = preparePointCoords(param);
    CHECK_IF_ERROR_RETURN(status, "Failed to prepare point coordinates");

    device::Device *cur_device = device::getDefaultHostDevice();
    device::TensorDesc ori_im_size_desc;
    ori_im_size_desc.data_format_ = base::kDataFormatN;
    ori_im_size_desc.data_type_ = base::dataTypeOf<float>();
    ori_im_size_desc.shape_ = {2};
    device::Tensor *ori_im_size_tensor =
        outputs_[2]->create(cur_device, ori_im_size_desc);
    CHECK_IF_NULL_RETURN(ori_im_size_tensor,
                         "Failed to create ori_im_size_tensor");
    float *ori_im_size_data = (float *)ori_im_size_tensor->getData();
    ori_im_size_data[0] = static_cast<float>(param->ori_height);
    ori_im_size_data[1] = static_cast<float>(param->ori_width);

    return base::kStatusCodeOk;
  }

 private:
  base::Status preparePointCoords(const SAMPointsParam *param) {
    if (param->ori_width <= 0 || param->ori_height <= 0) {
      NNDEPLOY_LOGE("Invalid original image size: (%d, %d)", param->ori_width,
                    param->ori_height);
      return base::kStatusCodeErrorInvalidValue;
    }
    if (param->points_.empty() || param->labels_.empty()) {
      NNDEPLOY_LOGE("Points or labels are empty.");
      return base::kStatusCodeErrorInvalidValue;
    }

    if (param->points_.size() / 2 != param->labels_.size()) {
      NNDEPLOY_LOGE("Points and labels size mismatch: %zu vs %zu",
                    param->points_.size(), param->labels_.size());
      return base::kStatusCodeErrorInvalidValue;
    }

    int points_num = (int)param->points_.size() / 2;
    device::Device *cur_device = device::getDefaultHostDevice();

    device::TensorDesc point_coords_desc;
    point_coords_desc.data_format_ = base::kDataFormatNCL;
    point_coords_desc.data_type_ = base::dataTypeOf<float>();
    point_coords_desc.shape_ = {1, points_num, 2};

    device::Tensor *point_coords_tensor =
        outputs_[0]->create(cur_device, point_coords_desc);
    CHECK_IF_NULL_RETURN(point_coords_tensor,
                         "Failed to create point_coords_tensor");
    float *point_coords_data = (float *)point_coords_tensor->getData();

    device::TensorDesc point_labels_desc;
    point_labels_desc.data_format_ = base::kDataFormatNC;
    point_labels_desc.data_type_ = base::dataTypeOf<float>();
    point_labels_desc.shape_ = {1, points_num};
    device::Tensor *point_labels_tensor =
        outputs_[1]->create(cur_device, point_labels_desc);
    CHECK_IF_NULL_RETURN(point_labels_tensor,
                         "Failed to create point_labels_tensor");
    float *point_labels_data = (float *)point_labels_tensor->getData();

    const int model_size = 1024;
    int origin_h = param->ori_height;
    int origin_w = param->ori_width;
    float scale_h = (float)model_size / origin_h;
    float scale_w = (float)model_size / origin_w;
    int new_h, new_w;
    if (scale_h < scale_w) {
      new_w = std::round(origin_w * scale_h);
      new_h = model_size;
    } else {
      new_h = std::round(origin_h * scale_w);
      new_w = model_size;
    }
    float scale_width = (float)new_w / origin_w;
    float scale_height = (float)new_h / origin_h;

    for (int i = 0; i < points_num; ++i) {
      point_coords_data[2 * i + 0] = param->points_[2 * i + 0] * scale_width;
      point_coords_data[2 * i + 1] = param->points_[2 * i + 1] * scale_height;
      point_labels_data[i] = param->labels_[i];
    }
    return base::kStatusCodeOk;
  }
};

class SAMPostProcess : public dag::Node {
 public:
  SAMPostProcess(const std::string &name) : dag::Node(name) {
    key_ = "nndeploy::segment::SAMPostProcess";
    desc_ = "Segment Anything Post Process Node for image segmentation tasks.";
    this->setInputTypeInfo<device::Tensor>();
    this->setInputTypeInfo<device::Tensor>();
    this->setInputTypeInfo<device::Tensor>();
    this->setOutputTypeInfo<cv::Mat>();
  }
  SAMPostProcess(const std::string &name, std::vector<dag::Edge *> inputs,
                 std::vector<dag::Edge *> outputs)
      : dag::Node(name, inputs, outputs) {
    key_ = "nndeploy::segment::SAMPostProcess";
    desc_ = "Segment Anything Post Process Node for image segmentation tasks.";
    this->setInputTypeInfo<device::Tensor>();
    this->setInputTypeInfo<device::Tensor>();
    this->setInputTypeInfo<device::Tensor>();
    this->setOutputTypeInfo<cv::Mat>();
  }
  virtual ~SAMPostProcess() {}

  base::Status run() override {
    // int a = this->inputs_.size();

    device::Tensor *masks_tensor = inputs_[0]->getTensor(this);
    device::Tensor *iou_tensor = inputs_[1]->getTensor(this);
    // device::Tensor *c = inputs_[2]->getTensor(this);
    device::TensorDesc masks_desc = masks_tensor->getDesc();
    int result_num = masks_desc.shape_[1];
    float *masks_data = (float *)masks_tensor->getData();
    float *iou_data = (float *)iou_tensor->getData();

    int height = masks_desc.shape_[2];
    int width = masks_desc.shape_[3];
    int offset = height * width;

    int best_idx = 0;
    float best_iou = 0.f;
    for (int i = 0; i < result_num; ++i) {
      // NNDEPLOY_LOGI("iou: %f\n", iou_data[i]);
      if (iou_data[i] > best_iou) {
        best_iou = iou_data[i];
        best_idx = i;
      }
    }

    cv::Mat *result = new cv::Mat();
    cv::Mat mask(height, width, CV_32FC1, masks_data + best_idx * offset);
    mask.convertTo(*result, CV_8UC1, 255.0f);

    outputs_[0]->set(result, false);

    return base::kStatusCodeOk;
  }
};

/**
 * 当前暂不支持mask输入，保留Node方便后续扩展
 */
class SAMMaskNode : public dag::Node {
 public:
  SAMMaskNode(const std::string &name) : dag::Node(name) {
    key_ = "nndeploy::segment::SAMMaskNode";
    desc_ = "Segment Anything Mask Node for image segmentation tasks.";
    this->setOutputTypeInfo<device::Tensor>();
    this->setOutputTypeInfo<device::Tensor>();
    node_type_ = dag::NodeType::kNodeTypeInput;
  }
  SAMMaskNode(const std::string &name, std::vector<dag::Edge *> inputs,
              std::vector<dag::Edge *> outputs)
      : dag::Node(name, inputs, outputs) {
    key_ = "nndeploy::segment::SAMMaskNode";
    desc_ = "Segment Anything Mask Node for image segmentation tasks.";
    this->setOutputTypeInfo<device::Tensor>();
    this->setOutputTypeInfo<device::Tensor>();
    node_type_ = dag::NodeType::kNodeTypeInput;
  }
  virtual ~SAMMaskNode() {}

  base::Status run() override {
    device::Device *cur_device = device::getDefaultHostDevice();

    device::TensorDesc mask_desc;
    mask_desc.data_format_ = base::kDataFormatNCHW;
    mask_desc.data_type_ = base::dataTypeOf<float>();
    mask_desc.shape_ = {1, 1, 256, 256};

    device::Tensor *mask_input_tensor =
        outputs_[0]->create(cur_device, mask_desc);
    CHECK_IF_NULL_RETURN(mask_input_tensor,
                         "Failed to create mask_input_tensor");
    memset(mask_input_tensor->getData(), 0, mask_input_tensor->getSize());

    device::TensorDesc has_mask_desc;
    has_mask_desc.data_format_ = base::kDataFormatN;
    has_mask_desc.data_type_ = base::dataTypeOf<float>();
    has_mask_desc.shape_ = {1};
    device::Tensor *has_mask_tensor =
        outputs_[1]->create(cur_device, has_mask_desc);
    CHECK_IF_NULL_RETURN(has_mask_tensor, "Failed to create has_mask_tensor");
    memset(has_mask_tensor->getData(), 0, has_mask_tensor->getSize());

    return base::kStatusCodeOk;
  }
};

// base::Status SAMGraph::init() {
//   base::Status status = base::kStatusCodeOk;

//   status = initGraphNodes();
//   if (status != base::kStatusCodeOk) {
//     NNDEPLOY_LOGE("Failed to initialize graph nodes: %s",
//                   status.desc().c_str());
//     return status;
//   }

//   status = this->dag::Graph::init();
//   if (status != base::kStatusCodeOk) {
//     NNDEPLOY_LOGE("Failed to initialize graph: %s", status.desc().c_str());
//     return status;
//   }

//   return status;
// }

base::Status SAMGraph::setInferParam(base::InferenceType inference_type,
                                     base::DeviceType device_type,
                                     base::ModelType model_type, bool is_path,
                                     std::vector<std::string> &model_value) {
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
  decoder_infer_param_.max_shape_.insert({"point_coords", {1, 3, 2}});
  decoder_infer_param_.max_shape_.insert({"point_labels", {1, 3, 1}});
  status = decoder_infer_node_->setInferenceType(
      decoder_infer_param_.inference_type_);
  CHECK_IF_ERROR_RETURN(status,
                        "Failed to set inference type for decoder_infer");
  status = decoder_infer_node_->setParam(&decoder_infer_param_);
  CHECK_IF_ERROR_RETURN(status, "Failed to set param for decoder_infer node");

  return status;
}

base::Status SAMGraph::initGraphNodes() {
  base::Status status = base::kStatusCodeOk;

  dag::Edge *encoder_input = this->createEdge("encoder_input");
  CHECK_IF_NULL_RETURN(encoder_input, "Failed to create encoder_input edge");
  dag::Edge *decoder_input = this->createEdge("decoder_input");
  CHECK_IF_NULL_RETURN(decoder_input, "Failed to create decoder_input edge");
  dag::Edge *mask_input = this->createEdge("mask_input");
  CHECK_IF_NULL_RETURN(mask_input, "Failed to create mask_input edge");
  dag::Edge *has_mask_input = this->createEdge("has_mask_input");
  CHECK_IF_NULL_RETURN(has_mask_input, "Failed to create has_mask_input edge");
  dag::Edge *point_coords = this->createEdge("point_coords");
  CHECK_IF_NULL_RETURN(point_coords, "Failed to create point_coords edge");
  dag::Edge *point_labels = this->createEdge("point_labels");
  CHECK_IF_NULL_RETURN(point_labels, "Failed to create point_labels edge");
  dag::Edge *ori_im_size = this->createEdge("orig_im_size");
  CHECK_IF_NULL_RETURN(ori_im_size, "Failed to create ori_im_size edge");
  dag::Edge *decoder_output_mask = this->createEdge("masks");
  CHECK_IF_NULL_RETURN(decoder_output_mask,
                       "Failed to create decoder_output_mask edge");
  dag::Edge *decoder_output_iou = this->createEdge("iou_predictions");
  CHECK_IF_NULL_RETURN(decoder_output_iou,
                       "Failed to create decoder_output_iou edge");
  dag::Edge *decoder_low_res_masks = this->createEdge("low_res_masks");
  CHECK_IF_NULL_RETURN(decoder_low_res_masks,
                       "Failed to create decoder_low_res_masks edge");

  preprocess_image_node_ = this->createNode<preprocess::CvtResizePadNormTrans>(
      "preprocess_image", {inputs_[0]}, {encoder_input});
  CHECK_IF_NULL_RETURN(preprocess_image_node_,
                       "Failed to create preprocess_image node");

  preprocess::CvtResizePadNormTransParam *preprocess_image_param =
      (preprocess::CvtResizePadNormTransParam *)
          preprocess_image_node_->getParam();
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

  encoder_infer_node_ = (infer::Infer *)this->createNode<infer::Infer>(
      "encoder_infer", {encoder_input}, {decoder_input});
  CHECK_IF_NULL_RETURN(encoder_infer_node_,
                       "Failed to create encoder_infer node");

  preprocess_point_node_ =
      this->createNode<SAMPointNode>("preprocess_point", {inputs_[1]},
                                     {point_coords, point_labels, ori_im_size});
  CHECK_IF_NULL_RETURN(preprocess_point_node_,
                       "Failed to create preprocess_point node");

  std::vector<dag::Edge *> no_input;
  std::vector<dag::Edge *> mask_output = {mask_input, has_mask_input};
  preprocess_mask_node_ =
      this->createNode<SAMMaskNode>("preprocess_mask", no_input, mask_output);
  CHECK_IF_NULL_RETURN(preprocess_mask_node_,
                       "Failed to create preprocess_mask node");

  decoder_infer_node_ = (infer::Infer *)this->createNode<infer::Infer>(
      "decoder_infer",
      {decoder_input, point_coords, point_labels, mask_input, has_mask_input,
       ori_im_size},
      {decoder_output_mask, decoder_output_iou, decoder_low_res_masks});
  CHECK_IF_NULL_RETURN(decoder_infer_node_,
                       "Failed to create decoder_infer node");

  postprocess_node_ = this->createNode<SAMPostProcess>(
      "postprocess",
      {decoder_output_mask, decoder_output_iou, decoder_low_res_masks},
      {outputs_[0]});
  CHECK_IF_NULL_RETURN(postprocess_node_, "Failed to create postprocess node");

  return status;
}
}  // namespace segment
}  // namespace nndeploy