#include "nndeploy/source/base/glic_stl_include.h"
#include "nndeploy/source/base/time_measurement.h"
#include "nndeploy/source/inference/inference_param.h"
#include "nndeploy/source/task/detect/opencv/post_process.h"
#include "nndeploy/source/task/pre_process/opencv/cvtcolor_resize.h"
#include "nndeploy/source/task/task.h"

using namespace nndeploy;

cv::Mat draw_box(cv::Mat &cv_mat, task::DetectResult &result) {
  float w_ratio = float(cv_mat.cols) / float(640);
  float h_ratio = float(cv_mat.rows) / float(640);
  int CNUM = 80;
  cv::RNG rng(0xFFFFFFFF);
  cv::Scalar_<int> randColor[CNUM];
  for (int i = 0; i < CNUM; i++)
    rng.fill(randColor[i], cv::RNG::UNIFORM, 0, 256);
  int i = -1;
  for (auto box : result.boxes_) {
    ++i;
    box[0] *= w_ratio;
    box[2] *= w_ratio;
    box[1] *= h_ratio;
    box[3] *= h_ratio;
    int width = box[2] - box[0];
    int height = box[3] - box[1];
    int id = result.label_ids_[i];
    cv::Point p = cv::Point(box[0], box[1]);
    cv::Rect rect = cv::Rect(box[0], box[1], width, height);
    cv::rectangle(cv_mat, rect, randColor[result.label_ids_[i]]);
    std::string text = " ID:" + std::to_string(id);
    cv::putText(cv_mat, text, cv::Point(0, 0), cv::FONT_HERSHEY_PLAIN, 1,
                randColor[result.label_ids_[i]]);
  }
  return cv_mat;
}

int yolo_main(int argc, char *argv[]) {
  base::TimeMeasurement *tm = new base::TimeMeasurement();

  base::DeviceType device_type(base::kDeviceTypeCodeX86, 0);
  task::Task *task =
      new task::Task(base::kInferenceTypeMnn, device_type, "yolov5");
  task->createPreprocess<task::OpencvCvtColrResize>();
  task->createPostprocess<task::DetectPostProcess>();

  CvtclorResizeParam *pre_param =
      dynamic_cast<CvtclorResizeParam *>(task->getPreProcessParam());
  pre_param->src_pixel_type_ = base::kPixelTypeBGR;
  pre_param->dst_pixel_type_ = base::kPixelTypeBGR;
  pre_param->interp_type_ = base::kInterpTypeLinear;
  pre_param->mean_[0] = 0.0f;
  pre_param->mean_[1] = 0.0f;
  pre_param->mean_[2] = 0.0f;
  pre_param->mean_[3] = 0.0f;
  pre_param->std_[0] = 255.0f;
  pre_param->std_[1] = 255.0f;
  pre_param->std_[2] = 255.0f;
  pre_param->std_[3] = 255.0f;
  inference::InferenceParam *inference_param =
      (inference::InferenceParam *)(task->getInferenceParam());
  inference_param->is_path_ = true;
  inference_param->model_value_.push_back("/home/always/Downloads/yolov5s.mnn");

  task->init();

  cv::Mat input_mat = cv::imread("/home/always/Downloads/yolo_input.jpeg");
  task::Packet input(input_mat);
  task::DetectResult result;
  task::Packet output(result);
  task->setInput(input);
  task->setOutput(output);
  task->run();

  draw_box(input_mat, result);
  cv::imwrite("/home/always/Downloads/yolo_result.jpg", input_mat);
  task->deinit();

  printf("hello world!\n");
  return 0;
}

int detr_main(int argc, char *argv[]) {
  base::TimeMeasurement *tm = new base::TimeMeasurement();

  base::DeviceType device_type(base::kDeviceTypeCodeX86, 0);
  task::Task *task =
      new task::Task(base::kInferenceTypeMnn, device_type, "detr");
  task->createPreprocess<task::OpencvCvtColrResize>();
  task->createPostprocess<task::DETRPostProcess>();

  CvtclorResizeParam *pre_param =
      dynamic_cast<CvtclorResizeParam *>(task->getPreProcessParam());
  pre_param->src_pixel_type_ = base::kPixelTypeBGR;
  pre_param->dst_pixel_type_ = base::kPixelTypeBGR;
  pre_param->interp_type_ = base::kInterpTypeLinear;
  pre_param->mean_[0] = 0.0f;
  pre_param->mean_[1] = 0.0f;
  pre_param->mean_[2] = 0.0f;
  pre_param->mean_[3] = 0.0f;
  pre_param->std_[0] = 255.0f;
  pre_param->std_[1] = 255.0f;
  pre_param->std_[2] = 255.0f;
  pre_param->std_[3] = 255.0f;
  inference::InferenceParam *inference_param =
      (inference::InferenceParam *)(task->getInferenceParam());
  inference_param->is_path_ = true;
  inference_param->model_value_.push_back("/home/always/Downloads/detr.mnn");

  task->init();

  cv::Mat input_mat = cv::imread("/home/always/Downloads/yolo_input.jpeg");
  task::Packet input(input_mat);
  task::DetectResult result;
  task::Packet output(result);
  task->setInput(input);
  task->setOutput(output);
  task->run();

  draw_box(input_mat, result);
  cv::imwrite("/home/always/Downloads/yolo_result.jpg", input_mat);
  task->deinit();

  printf("hello world!\n");
  return 0;
}

int main(int argc, char *argv[]) { yolo_main(argc, argv); }