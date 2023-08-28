#include "nndeploy/model/detect/yolo/yolov5.h"

#include "nndeploy/base/common.h"
#include "nndeploy/base/glic_stl_include.h"
#include "nndeploy/base/log.h"
#include "nndeploy/base/macro.h"
#include "nndeploy/base/object.h"
#include "nndeploy/base/opencv_include.h"
#include "nndeploy/base/shape.h"
#include "nndeploy/base/status.h"
#include "nndeploy/base/string.h"
#include "nndeploy/base/value.h"
#include "nndeploy/device/buffer.h"
#include "nndeploy/device/buffer_pool.h"
#include "nndeploy/device/device.h"
#include "nndeploy/device/tensor.h"
#include "nndeploy/model/packet.h"
#include "nndeploy/model/preprocess/cvtcolor_resize.h"
#include "nndeploy/model/task.h"

/*
TODO:
1.softmax在这里还有用吗？
*/

namespace nndeploy {
namespace model {

//  static TypeTaskRegister g_internal_opencv_detr_task_register("opencv_detr",
//                                                              creatDetrTask);//需要改

base::Status Yolov5PostProcess::run() {
  results_.bboxs_.clear();  // 先清空现有的mat，万一里面有东西

  Yolov5PostParam* temp_param = static_cast<Yolov5PostParam*>(param_.get());
  float score_threshold = temp_param->score_threshold_;
  Packet* input = inputs_[0];
  device::Tensor* input_tensor = input->getTensor();
  auto input_shape = input_tensor->getShape();
  auto output_tensor_0 = outputs_[0]->getTensor("output");
  auto output_tensor_1 = outputs_[0]->getTensor("463");
  auto output_tensor_2 = outputs_[0]->getTensor("482");

  GenerateDetectResult({output_tensor_0, output_tensor_1, output_tensor_2},
                       input_shape[3], input_shape[2]);

  return base::kStatusCodeOk;
}

// createTask怎么没了？
Pipeline* creatYoloPipeline(const std::string& name, base::InferenceType type,
                            Packet* input, Packet* output, bool is_path,
                            std::vector<std::string>& model_value) {
  Pipeline* pipeline = new Pipeline(name, input, output);  // 创建pipeline

  Packet* infer_input = pipeline->createPacket(
      "infer_input");  // 用pipeline创建空的packet：infer_input和infer_output:这时候还都是空的
  Packet* infer_output = pipeline->createPacket("infer_output");

  Task* pre =
      pipeline->createTask<CvtColrResize>(  // 前处理：input->infer_input
          "pre", input, infer_input);

  Task* infer =
      pipeline->createInfer<Infer>(  // infer：infer_input->infer_output
          "infer", type, infer_input, infer_output);

  Task* post = pipeline->createTask<Yolov5PostProcess>(
      "post", infer_output, output);  // 后处理：infer_output->output

  // 取出param，改变mean和std以及一大堆参数
  CvtclorResizeParam* pre_param =
      dynamic_cast<CvtclorResizeParam*>(pre->getParam());
  // 暂时找不到，没改 -去哪里找？这是不是个问题？
  pre_param->src_pixel_type_ = base::kPixelTypeRGB;  // nchw一般对应rgb？
  pre_param->dst_pixel_type_ = base::kPixelTypeRGB;
  pre_param->interp_type_ = base::kInterpTypeLinear;
  // scale和bias应该是不用改的

  inference::InferenceParam* inference_param =
      (inference::InferenceParam*)(infer->getParam());
  inference_param->is_path_ = is_path;
  inference_param->model_value_ = model_value;
  inference_param->device_type_ = device::getDefaultHostDeviceType();

  return pipeline;
}

void Yolov5PostProcess::PostProcessTensors(
    std::vector<std::shared_ptr<device::Tensor>> outputs,
    std::vector<std::shared_ptr<device::Tensor>>& post_tensor) {
  for (auto& output : outputs) {
    auto dims = output->getShape();  // dims不知道是不是一定是shape
    auto h_stride = base::shapeCount(dims, 2);  // 这里要改
    auto w_stride = base::shapeCount(dims, 3);
    base::IntVector permute_dims = {dims[0], dims[2], dims[3],
                                    dims[1] * dims[4]};

    device::TensorDesc tensor_desc = output->getDesc();
    tensor_desc.shape_ = permute_dims;
    auto tensor =
        std::make_shared<device::Tensor>(tensor_desc, output->getName());

    float* src_data = reinterpret_cast<float*>(output->getPtr());
    float* dst_data = reinterpret_cast<float*>(tensor->getPtr());

    int out_idx = 0;
    for (int h = 0; h < permute_dims[1]; h++) {
      for (int w = 0; w < permute_dims[2]; w++) {
        for (int s = 0; s < permute_dims[3]; s++) {
          // 在循环里做一坨计算，实际上做计算的地方
          int anchor_idx = s / dims[4];  // dim[1]
          int detect_idx = s % dims[4];
          int in_idx = anchor_idx * h_stride + h * w_stride + w * dims[4] +
                       detect_idx;  // 求出id
          dst_data[out_idx++] = 1.0f / (1.0f + exp(-src_data[in_idx]));
        }
      }
    }

    post_tensor.emplace_back(tensor);
  }
}

void Yolov5PostProcess::GenerateDetectResult(
    std::vector<std::shared_ptr<device::Tensor>> outputs, int image_width,
    int image_height) {
  int blob_index = 0;
  // 这里用post_process，得到mat
  std::vector<std::shared_ptr<device::Tensor>> post_Tensors;
  PostProcessTensors(outputs, post_Tensors);
  auto output_Tensors = post_Tensors;
  Yolov5PostParam* temp_param = static_cast<Yolov5PostParam*>(param_.get());
  for (auto& output : output_Tensors) {
    auto dim = output->getShape();

    if (dim[3] != temp_param->num_anchor_ * temp_param->detect_dim_) {
      NNDEPLOY_LOGE(
          "Invalid detect output, the size of last dimension is: %d\n", dim[3]);
      return;
    }
    float* data = static_cast<float*>(output->getPtr());

    int num_potential_detecs =
        dim[1] * dim[2] * temp_param->num_anchor_;  // 还是49个框的那一套
    for (int i = 0; i < num_potential_detecs; ++i) {
      // 分别取出每个框的x,y,w,h
      float x = data[i * temp_param->detect_dim_ + 0];
      float y = data[i * temp_param->detect_dim_ + 1];
      float width = data[i * temp_param->detect_dim_ + 2];
      float height = data[i * temp_param->detect_dim_ + 3];

      float objectness = data[i * temp_param->detect_dim_ + 4];  // 类别概率
      if (objectness < temp_param->conf_thres_)  // 筛选框的阈值
        continue;
      // center point coord
      x = (x * 2 - 0.5 + ((i / temp_param->num_anchor_) % dim[2])) *
          temp_param->strides_[blob_index];
      y = (y * 2 - 0.5 + ((i / temp_param->num_anchor_) / dim[2]) % dim[1]) *
          temp_param->strides_[blob_index];
      width =
          pow((width * 2), 2) *
          temp_param->anchor_grids_[blob_index * temp_param->grid_per_input_ +
                                    (i % temp_param->num_anchor_) * 2 + 0];
      height =
          pow((height * 2), 2) *
          temp_param->anchor_grids_[blob_index * temp_param->grid_per_input_ +
                                    (i % temp_param->num_anchor_) * 2 + 1];
      // compute coords
      float x1 = x - width / 2;
      float y1 = y - height / 2;
      float x2 = x + width / 2;
      float y2 = y + height / 2;
      // compute confidence
      auto conf_start = data + i * temp_param->detect_dim_ + 5;
      auto conf_end = data + (i + 1) * temp_param->detect_dim_;
      auto max_conf_iter = std::max_element(conf_start, conf_end);
      int conf_idx = static_cast<int>(std::distance(conf_start, max_conf_iter));
      float score = (*max_conf_iter) * objectness;

      // obj_info.image_width = image_width;
      // obj_info.image_height = image_height;
      // index不知道是什么，没填上
      results_.bboxs_[i].bbox_ = {x1, y1, x2, y2};
      results_.bboxs_[i].score_ = score;
      results_.bboxs_[i].label_id_ = conf_idx;
      // results_.result_.push_back(obj_info);
    }
    blob_index += 1;
  }

  DetectResult temp_results_;
  std::vector<int> keep_idxs;

  computeNMS(results_.bboxs_,  // 这些命名空间都不对，要改！
             temp_param->iou_thres_, temp_results_);
  results_.emplace_back(temp_results_);  // 不知道emplace_back这样写对不对
}

}  // namespace model
}  // namespace nndeploy
