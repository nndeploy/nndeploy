
#include "nndeploy/segment/rmbg/rmbg.h"

#include "nndeploy/base/any.h"
#include "nndeploy/base/common.h"
#include "nndeploy/base/glic_stl_include.h"
#include "nndeploy/base/log.h"
#include "nndeploy/base/macro.h"
#include "nndeploy/base/object.h"
#include "nndeploy/base/opencv_include.h"
#include "nndeploy/base/status.h"
#include "nndeploy/base/string.h"
#include "nndeploy/dag/edge.h"
#include "nndeploy/dag/node.h"
#include "nndeploy/detect/util.h"
#include "nndeploy/device/buffer.h"
#include "nndeploy/device/device.h"
#include "nndeploy/device/memory_pool.h"
#include "nndeploy/device/tensor.h"
#include "nndeploy/infer/infer.h"
#include "nndeploy/preprocess/cvt_resize_norm_trans.h"

namespace nndeploy {
namespace segment {

base::Status RMBGPostParam::serialize(rapidjson::Value &json,
                                 rapidjson::Document::AllocatorType &allocator){
  rapidjson::Value key_value(rapidjson::kObjectType);
  key_value.AddMember("version", version_, allocator);
  return base::kStatusCodeOk;
}

base::Status RMBGPostParam::deserialize(rapidjson::Value &json) {
  if (!json.HasMember("version_") || !json["version_"].IsInt()) {
    return base::kStatusCodeErrorInvalidValue;
  }
  version_ = json["version_"].GetInt();
  return base::kStatusCodeOk;
}


base::Status RMBGPostProcess::run() {
  // 从输入边缘获取输入图像矩阵
  cv::Mat *input_mat = inputs_[0]->getCvMat(this);
  // 获取输入图像的高度和宽度
  int dst_height = input_mat->rows;
  int dst_width = input_mat->cols;
  // 从第二个输入边缘获取张量数据
  device::Tensor *tensor = inputs_[1]->getTensor(this);
  // 获取张量数据的指针
  float *data = (float *)tensor->getData();
  // 获取张量的批次、通道、高度和宽度信息
  int batch = tensor->getBatch();
  int channel = tensor->getChannel();
  int height = tensor->getHeight();
  int width = tensor->getWidth();

  // 创建一个临时的单通道浮点矩阵用于存放张量数据
  cv::Mat temp(height, width, CV_32FC1, data);
  cv::Mat mask;
  // 将临时矩阵调整大小以匹配输入图像的尺寸
  cv::resize(temp, mask, input_mat->size(), 0.0, 0.0, cv::INTER_LINEAR);
  // 计算调整大小后的矩阵的最小值和最大值
  double minVal, maxVal;
  cv::minMaxLoc(mask, &minVal, &maxVal);

  // 将矩阵归一化到0到1之间
  cv::Mat normalized = (mask - minVal) / (maxVal - minVal);

  // 创建结果对象并初始化
  SegmentResult *results = new SegmentResult();

  // 设置输出张量的描述信息
  device::TensorDesc desc;
  desc.data_type_ = base::dataTypeOf<uint8_t>();
  desc.data_format_ = base::kDataFormatNHWC;
  desc.shape_ = {1, dst_height, dst_width, 1};
  // 获取设备对象
  device::Device *device = tensor->getDevice();
  // 创建输出张量
  device::Tensor *dst = new device::Tensor(device, desc);
  // 设置结果对象的掩码张量
  results->mask_ = dst;
  results->score_ = nullptr;
  results->height_ = dst_height;
  results->width_ = dst_width;
  results->classes_ = -1;

  // 创建目标图像矩阵
  cv::Mat dst_mat(dst_height, dst_width, CV_8UC1, dst->getData());

  // 将归一化后的矩阵转换为8位无符号整数格式
  normalized.convertTo(dst_mat, CV_8UC1, 255.0);

  // 将结果设置到输出边缘
  outputs_[0]->set(results, false);

  // 返回操作成功状态
  return base::kStatusCodeOk;
}

// dag::Graph *createRMBGGraph(const std::string &name,
//                             base::InferenceType inference_type,
//                             base::DeviceType device_type, dag::Edge *input,
//                             dag::Edge *output, base::ModelType model_type,
//                             bool is_path,
//                             std::vector<std::string> model_value) {
//   dag::Graph *graph = new dag::Graph(name, {input}, {output});
//   dag::Edge *infer_input = graph->createEdge("input");
//   dag::Edge *infer_output = graph->createEdge("output");

//   dag::Node *pre = graph->createNode<preprocess::CvtResizeNormTrans>(
//       "preprocess", {input}, {infer_input});

//   infer::Infer *infer = dynamic_cast<infer::Infer *>(
//       graph->createNode<infer::Infer>("infer", {infer_input},
//       {infer_output}));
//   if (infer == nullptr) {
//     NNDEPLOY_LOGE("Failed to create inference node");
//     return nullptr;
//   }
//   infer->setInferenceType(inference_type);

//   dag::Node *post = graph->createNode<RMBGPostProcess>(
//       "postprocess", {input, infer_output}, {output});

//   preprocess::CvtResizeNormTransParam *pre_param =
//       dynamic_cast<preprocess::CvtResizeNormTransParam *>(pre->getParam());
//   pre_param->src_pixel_type_ = base::kPixelTypeBGR;
//   pre_param->dst_pixel_type_ = base::kPixelTypeRGB;
//   pre_param->interp_type_ = base::kInterpTypeLinear;
//   pre_param->h_ = 1024;
//   pre_param->w_ = 1024;
//   pre_param->mean_[0] = 0.5f;
//   pre_param->mean_[1] = 0.5f;
//   pre_param->mean_[2] = 0.5f;
//   pre_param->mean_[3] = 0.5f;

//   inference::InferenceParam *inference_param =
//       (inference::InferenceParam *)(infer->getParam());
//   inference_param->is_path_ = is_path;
//   inference_param->model_value_ = model_value;
//   inference_param->device_type_ = device_type;
//   inference_param->model_type_ = model_type;

//   // TODO: 很多信息可以从 preprocess 和 infer 中获取
//   RMBGPostParam *post_param = dynamic_cast<RMBGPostParam
//   *>(post->getParam()); post_param->version_ = 14;

//   return graph;
// }

REGISTER_NODE("nndeploy::segment::RMBGPostProcess", RMBGPostProcess);
REGISTER_NODE("nndeploy::segment::SegmentRMBGGraph", SegmentRMBGGraph);

}  // namespace segment
}  // namespace nndeploy
