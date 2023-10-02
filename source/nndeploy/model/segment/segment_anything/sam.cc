#include "nndeploy/model/segment/segment_anything/sam.h"
#include "nndeploy/base/opencv_include.h"
#include "nndeploy/model/preprocess/cvtcolor_resize.h"
#include "nndeploy/model/segment/util.h"

namespace nndeploy {
namespace model {

TypePipelineRegister g_register_sam_pipeline(NNDEPLOY_SAM, createSamPipeline);

// 后处理
base::Status SamPostProcess::run() {
  SamPostParam* param = dynamic_cast<SamPostParam*>(param_.get());
}

// 构建SAM的input
base::Status SamBuildInput::run() {
  int length = 1024;  // TODO(sjx): 修改为配置项
  int new_h, new_w;
  device::Device* cur_device = device::getDefaultHostDevice();

  int origin_h = 768;
  int origin_w = 756;
  if (origin_h > origin_w) {
    new_w = round(origin_w * (float)length / origin_h);
    new_h = length;
  } else {
    new_h = round(origin_h * (float)length / origin_w);
    new_w = length;
  }

  float scale_w = (float)new_w / origin_w;
  float scale_h = (float)new_h / origin_h;

  std::vector<float> points = {500, 375};  // TODO(sjx): 修改为配置项
  auto scale_points = points;
  for (int i = 0; i < scale_points.size() / 2; i++) {
    scale_points[2 * i] = scale_points[2 * i] * scale_w;
    scale_points[2 * i + 1] = scale_points[2 * i + 1] * scale_h;
  }
  scale_points.push_back(0);
  scale_points.push_back(0);

  // point_coords
  device::Tensor* point_coords_tensor =
      convertVectorToTensor(scale_points, {1, 2, 2}, cur_device,
                            base::kDataFormatNCHW, "point_coords");

  // point_labels
  std::vector<float> point_labels = {1, -1};

  device::Tensor* point_labels_tensor = convertVectorToTensor(
      point_labels, {1, 2}, cur_device, base::kDataFormatNCHW, "point_labels");

  // has_mask_input
  std::vector<float> has_mask_input = {1};
  device::Tensor* has_mask_input_tensor = convertVectorToTensor(
      has_mask_input, {1}, cur_device, base::kDataFormatNCHW, "has_mask_input");

  // mask_input
  std::vector<float> mask_input(size_t(256 * 256), float(0));
  device::Tensor* mask_input_tensor =
      convertVectorToTensor(mask_input, {1, 1, 256, 256}, cur_device,
                            base::kDataFormatNCHW, "mask_input");

  // orig_im_size
  std::vector<float> orig_im_size = {float(origin_h), float(origin_w)};
  device::Tensor* orig_im_size_tensor = convertVectorToTensor(
      orig_im_size, {2}, cur_device, base::kDataFormatNCHW, "orig_im_size");

  outputs_[0]->set(has_mask_input_tensor, 0);
  outputs_[0]->set(inputs_[0]->getTensor(), 1);
  outputs_[0]->set(mask_input_tensor, 2);
  outputs_[0]->set(orig_im_size_tensor, 3);
  outputs_[0]->set(point_coords_tensor, 4);
  outputs_[0]->set(point_labels_tensor, 5);

  return base::kStatusCodeOk;
}

model::Pipeline* createSamPipeline(const std::string& name,
                                   base::InferenceType inference_type,
                                   base::DeviceType device_type, Packet* input,
                                   Packet* output, base::ModelType model_type,
                                   bool is_path,
                                   std::vector<std::string> model_values) {
  model::Pipeline* pipeline = new model::Pipeline(name, input, output);
  model::Packet* embedding_input = pipeline->createPacket("embedding_input");
  model::Packet* embedding_output = pipeline->createPacket("embedding_output");
  model::Packet* build_sam_input = pipeline->createPacket("build_input");
  model::Packet* segment_output = pipeline->createPacket("segment_output");

  model::Task* preprocess = pipeline->createTask<model::CvtColrResize>(
      "preprocess", input, embedding_input);

  model::Task* embedding_inference = pipeline->createInfer<model::Infer>(
      "embedding_inference", inference_type, embedding_input, embedding_output);

  model::Task* build_input = pipeline->createTask<SamBuildInput>(
      "build_input", embedding_output, build_sam_input);

  model::Task* segment_inference = pipeline->createInfer<model::Infer>(
      "segment_inference", inference_type, build_sam_input, segment_output);

  model::Task* postprocess = pipeline->createTask<SamPostProcess>(
      "postprocess", segment_output, output);


  model::CvtclorResizeParam* pre_param =
      dynamic_cast<model::CvtclorResizeParam*>(preprocess->getParam());
  pre_param->src_pixel_type_ = base::kPixelTypeBGR;
  pre_param->dst_pixel_type_ = base::kPixelTypeRGB;
  pre_param->interp_type_ = base::kInterpTypeLinear;
  pre_param->h_ = 1024;
  pre_param->w_ = 1024;  
  pre_param->mean_[1]=123.675;
  pre_param->mean_[2]=116.28;
  pre_param->mean_[3]=103.53;
  pre_param->std_[1]=58.395;
  pre_param->std_[2]=57.12;
  pre_param->std_[3]=57.375;


  inference::InferenceParam* embedding_inference_param =
      (inference::InferenceParam*)(embedding_inference->getParam());


// TODO： 由于sam包含两个模型，当前flag只支持传一个地址进来，因此先在这里写死两个模型地址，后续改为传参
  std::string path1 =
      "/data/sjx/code/nndeploy_resource/nndeploy/model_zoo/segment/sam/image_encoder_sim.onnx";
  std::string path2 = "/data/sjx/code/segment-anything/sam.onnx";

  embedding_inference_param->is_path_ = is_path;
  embedding_inference_param->model_value_ = std::vector<std::string>(1, path1);
  embedding_inference_param->device_type_ = device_type;

  inference::InferenceParam* segment_inference_param =
      (inference::InferenceParam*)(segment_inference->getParam());

  segment_inference_param->is_path_ = is_path;
  segment_inference_param->model_value_ = std::vector<std::string>(1, path2);
  segment_inference_param->device_type_ = device_type;




  SamPostParam* post_param =
      dynamic_cast<SamPostParam*>(postprocess->getParam());

  return pipeline;
}

}  // namespace model
}  // namespace nndeploy