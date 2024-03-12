#include "nndeploy/model/segment/segment_anything/sam.h"

#include "nndeploy/base/opencv_include.h"
#include "nndeploy/model/infer.h"
#include "nndeploy/model/preprocess/cvtcolor_resize_pad.h"
#include "nndeploy/model/segment/util.h"

namespace nndeploy {
namespace model {

dag::TypeGraphRegister g_register_sam_graph(NNDEPLOY_SAM, createSamGraph);

// 后处理
base::Status SamPostProcess::run() {
  SamPostParam *param = dynamic_cast<SamPostParam *>(param_.get());

  device::Tensor *mask = inputs_[0]->getTensor(this);
  SegmentResult *results = new SegmentResult();

  results->mask_ = mask;
  outputs_[0]->set(results, inputs_[0]->getIndex(this), false);
  return base::kStatusCodeOk;
}

base::Status SamBuildInput::run() {
  device::Device *cur_device = device::getDefaultHostDevice();

  cv::Mat *src = inputs_[0]->getCvMat(this);
  int origin_h = src->rows;
  int origin_w = src->cols;
  float scale_h = (float)1024 / origin_h;
  float scale_w = (float)1024 / origin_w;
  int new_h, new_w;
  if (scale_h < scale_w) {
    new_w = std::round(origin_w * scale_h);
    new_h = 1024;
  } else {
    new_h = std::round(origin_h * scale_w);
    new_w = 1024;
  }
  for (auto iter : this->outputs_) {
    std::string name = iter->getName();
    device::Device *cur_device = device::getDefaultHostDevice();
    device::TensorDesc desc;
    if (name == "point_coords") {
      float scale_w = (float)new_w / origin_w;
      float scale_h = (float)new_h / origin_h;
      std::vector<float> points = {500, 375};  // TODO(sjx): 修改为配置项
      auto scale_points = points;
      for (int i = 0; i < scale_points.size() / 2; i++) {
        scale_points[2 * i] = scale_points[2 * i] * scale_w;
        scale_points[2 * i + 1] = scale_points[2 * i + 1] * scale_h;
      }
      scale_points.emplace_back(0);
      scale_points.emplace_back(0);
      desc.data_type_ = base::dataTypeOf<float>();
      desc.data_format_ = base::kDataFormatNCW;
      desc.shape_.emplace_back(1);
      desc.shape_.emplace_back(2);
      desc.shape_.emplace_back(2);
      device::Tensor *point_coords_tensor =
          iter->create(cur_device, desc, inputs_[0]->getIndex(this));

      size_t size = point_coords_tensor->getSize();
      void *data = point_coords_tensor->getPtr();
      std::memcpy(data, (void *)scale_points.data(), size);
    } else if (name == "point_labels") {
      std::vector<float> point_labels = {1, -1};
      desc.data_type_ = base::dataTypeOf<float>();
      desc.data_format_ = base::kDataFormatNCW;
      desc.shape_.emplace_back(1);
      desc.shape_.emplace_back(1);
      device::Tensor *point_labels_tensor =
          iter->create(cur_device, desc, inputs_[0]->getIndex(this));
      size_t size = point_labels_tensor->getSize();
      void *data = point_labels_tensor->getPtr();
      std::memcpy(data, (void *)point_labels.data(), size);
    } else if (name == "has_mask_input") {
      std::vector<float> has_mask_input = {1};
      desc.data_type_ = base::dataTypeOf<float>();
      desc.data_format_ = base::kDataFormatN;
      desc.shape_.emplace_back(1);
      device::Tensor *has_mask_input_tensor =
          iter->create(cur_device, desc, inputs_[0]->getIndex(this));
      size_t size = has_mask_input_tensor->getSize();
      void *data = has_mask_input_tensor->getPtr();
      std::memcpy(data, (void *)has_mask_input.data(), size);
    } else if (name == "mask_input") {
      std::vector<float> mask_input(size_t(256 * 256), float(0));
      desc.data_type_ = base::dataTypeOf<float>();
      desc.data_format_ = base::kDataFormatNCHW;
      desc.shape_.emplace_back(1);
      desc.shape_.emplace_back(1);
      desc.shape_.emplace_back(256);
      desc.shape_.emplace_back(256);
      device::Tensor *mask_input_tensor =
          iter->create(cur_device, desc, inputs_[0]->getIndex(this));
      size_t size = mask_input_tensor->getSize();
      void *data = mask_input_tensor->getPtr();
      std::memcpy(data, (void *)mask_input.data(), size);
    } else if (name == "orig_im_size")

    {
      std::vector<float> orig_im_size = {float(origin_h), float(origin_w)};
      desc.data_type_ = base::dataTypeOf<float>();
      desc.data_format_ = base::kDataFormatN;
      desc.shape_.emplace_back(2);
      device::Tensor *orig_im_size_tensor =
          iter->create(cur_device, desc, inputs_[0]->getIndex(this));
      size_t size = orig_im_size_tensor->getSize();
      void *data = orig_im_size_tensor->getPtr();
      std::memcpy(data, (void *)orig_im_size.data(), size);
    }
  }
  return base::kStatusCodeOk;
}

dag::Graph *createSamGraph(const std::string &name,
                           base::InferenceType inference_type,
                           base::DeviceType device_type, dag::Edge *input,
                           dag::Edge *output, base::ModelType model_type,
                           bool is_path,
                           std::vector<std::string> model_values) {
  dag::Graph *graph = new dag::Graph(name, input, output);
  dag::Edge *input_image = graph->createEdge("input_image");
  dag::Edge *image_embeddings = graph->createEdge("image_embeddings");

  dag::Edge *segment_output = graph->createEdge("masks");

  dag::Edge *point_coords = graph->createEdge("point_coords");
  dag::Edge *point_labels = graph->createEdge("point_labels");
  dag::Edge *has_mask_input = graph->createEdge("has_mask_input");
  dag::Edge *mask_input = graph->createEdge("mask_input");
  dag::Edge *orig_im_size = graph->createEdge("orig_im_size");

  dag::Node *preprocess = graph->createNode<model::CvtColorResizePad>(
      "preprocess", input, input_image);
  dag::Node *embedding_inference = graph->createInfer<model::Infer>(
      "embedding_inference", inference_type, input_image, image_embeddings);

  dag::Node *build_input = graph->createNode<SamBuildInput>(
      "build_input", {input},
      {has_mask_input, point_coords, point_labels, orig_im_size, mask_input});
  // 这个推理任务本身是动态输入的
  dag::Node *segment_inference = graph->createInfer<model::Infer>(
      "segment_inference", inference_type,
      {image_embeddings, has_mask_input, point_coords, point_labels,
       orig_im_size, mask_input},
      {segment_output});
  dag::Node *postprocess =
      graph->createNode<SamPostProcess>("postprocess", segment_output, output);

  model::CvtclorResizePadParam *pre_param =
      dynamic_cast<model::CvtclorResizePadParam *>(preprocess->getParam());
  pre_param->src_pixel_type_ = base::kPixelTypeBGR;
  pre_param->dst_pixel_type_ = base::kPixelTypeRGB;
  pre_param->interp_type_ = base::kInterpTypeLinear;
  pre_param->h_ = 1024;
  pre_param->w_ = 1024;
  pre_param->scale_[1] = 1.0f;
  pre_param->scale_[2] = 1.0f;
  pre_param->scale_[3] = 1.0f;
  pre_param->mean_[1] = 123.675;
  pre_param->mean_[2] = 116.28;
  pre_param->mean_[3] = 103.53;
  pre_param->std_[1] = 58.395;
  pre_param->std_[2] = 57.12;
  pre_param->std_[3] = 57.375;

  inference::InferenceParam *embedding_inference_param =
      (inference::InferenceParam *)(embedding_inference->getParam());
  embedding_inference_param->is_path_ = is_path;
  embedding_inference_param->model_value_ =
      std::vector<std::string>(1, model_values[0]);
  embedding_inference_param->device_type_ = device_type;

  inference::InferenceParam *segment_inference_param =
      (inference::InferenceParam *)(segment_inference->getParam());
  segment_inference_param->is_path_ = is_path;
  segment_inference_param->model_value_ =
      std::vector<std::string>(1, model_values[1]);
  segment_inference_param->device_type_ = device_type;
  segment_inference_param->is_dynamic_shape_ = true;
  segment_inference_param->max_shape_.insert({"point_coords", {1, 2, 2}});
  segment_inference_param->max_shape_.insert({"point_labels", {1, 2}});
  segment_inference_param->max_shape_.insert({"has_mask_input", {1}});
  segment_inference_param->max_shape_.insert({"mask_input", {1, 1, 256, 256}});
  segment_inference_param->max_shape_.insert({"orig_im_size", {2}});
  segment_inference_param->max_shape_.insert(
      {"image_embeddings", {1, 256, 64, 64}});

  SamPostParam *post_param =
      dynamic_cast<SamPostParam *>(postprocess->getParam());

  return graph;
}

}  // namespace model
}  // namespace nndeploy