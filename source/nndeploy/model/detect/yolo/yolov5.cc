#include "nndeploy/model/detect/yolov5/yolov5.h"

#include "nndeploy/base/common.h"
#include "nndeploy/base/glic_stl_include.h"
#include "nndeploy/base/log.h"
#include "nndeploy/base/macro.h"
#include "nndeploy/base/object.h"
#include "nndeploy/base/opencv_include.h"
#include "nndeploy/base/status.h"
#include "nndeploy/base/string.h"
#include "nndeploy/base/value.h"
#include "nndeploy/device/buffer.h"
#include "nndeploy/device/buffer_pool.h"
#include "nndeploy/device/device.h"
#include "nndeploy/device/tensor.h"
#include "nndeploy/pipeline/packet.h"
#include "nndeploy/pipeline/preprocess/opencv/cvtcolor_resize.h"
#include "nndeploy/pipeline/task.h"

/*
TODO:
1.softmax在这里还有用吗？
*/

namespace nndeploy {
namespace model {
namespace opencv {

// static TypeTaskRegister g_internal_opencv_detr_task_register("opencv_detr",
//                                                              creatDetrTask);//需要改

template <typename T>
int softmax(const T* src, T* dst, int length) {
  T denominator{0};
  for (int i = 0; i < length; ++i) {
    dst[i] = std::exp(src[i]);
    denominator += dst[i];
  }
  for (int i = 0; i < length; ++i) {
    dst[i] /= denominator;
  }
  return 0;
}

base::Status Yolov5PostProcess::run() {
  results_.result_.clear();  // 先清空现有的mat，万一里面有东西

  Yolov5PostParam* temp_param = (Yolov5PostParam*)param_.get();
  float score_threshold = temp_param->score_threshold_;
  Packet* input = inputs_[0];
  device::Tensor input_tensor = input->getTensor();
  auto input_shape = input_tensor.getShape();
  auto output_tensor_0 = outputs_[0]->getTensor("output");
  auto output_tensor_1 = outputs_[0]->getTensor("463");
  auto output_tensor_2 = outputs_[0]->getTensor("482");

  GenerateDetectResult({output_tensor_0, output_tensor_1, output_tensor_2},
                       input_shape[3], input_shape[2]);

  return base::kStatusCodeOk;
}

// createTask怎么没了？
pipeline::Pipeline* creatYoloPipeline(const std::string& name,
                                      base::InferenceType type,
                                      pipeline::Packet* input,
                                      pipeline::Packet* output, bool is_path,
                                      std::vector<std::string>& model_value) {
  pipeline::Pipeline* pipeline =
      new pipeline::Pipeline(name, input, output);  // 创建pipeline

  pipeline::Packet* infer_input = pipeline->createPacket(
      "infer_input");  // 用pipeline创建空的packet：infer_input和infer_output:这时候还都是空的
  pipeline::Packet* infer_output = pipeline->createPacket("infer_output");

  pipeline::Task* pre = pipeline->createTask<
      pipeline::opencv::CvtColrResize>(  // 前处理：input->infer_input
      "pre", input, infer_input);

  pipeline::Task* infer =
      pipeline
          ->createInfer<pipeline::Infer>(  // infer：infer_input->infer_output
              "infer", type, infer_input, infer_output);

  pipeline::Task* post = pipeline->createTask<DetrPostProcess>(
      "post", infer_output, output);  // 后处理：infer_output->output

  // 取出param，改变mean和std以及一大堆参数
  pipeline::CvtclorResizeParam* pre_param =
      dynamic_cast<pipeline::CvtclorResizeParam*>(pre->getParam());
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

  void Yolov5PostProcess::GenerateDetectResult(
      std::vector<std::shared_ptr<device::Tensor>> outputs, int image_width,
      int image_height) {
    int blob_index = 0;
    // 这里用post_process，得到mat
    std::vector<std::shared_ptr<device::Tensor>> post_Tensors;
    PostProcessTensors(outputs, post_Tensors);
    auto output_Tensors = post_Tensors;

    for (auto& output : output_Tensors) {
      auto dim = output->getShape();

      if (dim[3] != num_anchor_ * detect_dim_) {
        LOGE("Invalid detect output, the size of last dimension is: %d\n",
             dim[3]);
        return;
      }
      float* data = static_cast<float*>(output->getPtr());

      int num_potential_detecs =
          dim[1] * dim[2] * num_anchor_;  // 还是49个框的那一套
      for (int i = 0; i < num_potential_detecs; ++i) {
        // 分别取出每个框的x,y,w,h
        float x = data[i * detect_dim_ + 0];
        float y = data[i * detect_dim_ + 1];
        float width = data[i * detect_dim_ + 2];
        float height = data[i * detect_dim_ + 3];

        float objectness = data[i * detect_dim_ + 4];  // 类别概率
        if (objectness < conf_thres)                   // 筛选框的阈值
          continue;
        // center point coord
        x = (x * 2 - 0.5 + ((i / num_anchor_) % dim[2])) * strides_[blob_index];
        y = (y * 2 - 0.5 + ((i / num_anchor_) / dim[2]) % dim[1]) *
            strides_[blob_index];
        width =
            pow((width * 2), 2) * anchor_grids_[blob_index * grid_per_input_ +
                                                (i % num_anchor_) * 2 + 0];
        height =
            pow((height * 2), 2) * anchor_grids_[blob_index * grid_per_input_ +
                                                 (i % num_anchor_) * 2 + 1];
        // compute coords
        float x1 = x - width / 2;
        float y1 = y - height / 2;
        float x2 = x + width / 2;
        float y2 = y + height / 2;
        // compute confidence
        auto conf_start = data + i * detect_dim_ + 5;
        auto conf_end = data + (i + 1) * detect_dim_;
        auto max_conf_iter = std::max_element(conf_start, conf_end);
        int conf_idx =
            static_cast<int>(std::distance(conf_start, max_conf_iter));
        float score = (*max_conf_iter) * objectness;

        // obj_info.image_width = image_width;
        // obj_info.image_height = image_height;
        // index不知道是什么，没填上
        results_.result_.bbox = {x1, y1, x2, y2};
        results_.result_.score_ = score;
        results_.result_.label_id = conf_idx;
        // results_.result_.push_back(obj_info);
      }
      blob_index += 1;
    }

    DetectResults temp_results_;
    std::vector<int> keep_idxs;

    computeNMS(results_.result_,  // 这些命名空间都不对，要改！
               iou_thres, temp_results_);
    results_.result_.emplace_back(
        temp_results_);  // 不知道emplace_back这样写对不对
  }

}  // namespace opencv
}  // namespace model
}  // namespace nndeploy
