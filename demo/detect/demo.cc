#include "flag.h"
#include "nndeploy/base/glic_stl_include.h"
#include "nndeploy/base/time_profiler.h"
#include "nndeploy/device/device.h"
#include "nndeploy/model/detect/yolo/yolo.h"
#include "nndeploy/model/task.h"

using namespace nndeploy;

cv::Mat draw_box(cv::Mat &cv_mat, model::DetectResult &result) {
  // float w_ratio = float(cv_mat.cols) / float(640);
  // float h_ratio = float(cv_mat.rows) / float(640);
  float w_ratio = float(cv_mat.cols);
  float h_ratio = float(cv_mat.rows);
  const int CNUM = 80;
  cv::RNG rng(0xFFFFFFFF);
  cv::Scalar_<int> randColor[CNUM];
  for (int i = 0; i < CNUM; i++)
    rng.fill(randColor[i], cv::RNG::UNIFORM, 0, 256);
  int i = -1;
  for (auto bbox : result.bboxs_) {
    std::array<float, 4> box;
    box[0] = bbox.bbox_[0];  // 640.0;
    box[2] = bbox.bbox_[2];  // 640.0;
    box[1] = bbox.bbox_[1];  // 640.0;
    box[3] = bbox.bbox_[3];  // 640.0;
    NNDEPLOY_LOGE("box[0]:%f, box[2]:%f, box[1]:%f, box[3]:%f\n", box[0],
                  box[2], box[1], box[3]);
    box[0] *= w_ratio;
    box[2] *= w_ratio;
    box[1] *= h_ratio;
    box[3] *= h_ratio;
    int width = box[2] - box[0];
    int height = box[3] - box[1];
    int id = bbox.label_id_;
    cv::Point p = cv::Point(box[0], box[1]);
    cv::Rect rect = cv::Rect(box[0], box[1], width, height);
    cv::rectangle(cv_mat, rect, randColor[id]);
    std::string text = " ID:" + std::to_string(id);
    cv::putText(cv_mat, text, p, cv::FONT_HERSHEY_PLAIN, 1, randColor[id]);
  }
  return cv_mat;
}
//
int main(int argc, char *argv[]) {
  gflags::ParseCommandLineNonHelpFlags(&argc, &argv, true);
  if (demo::FLAGS_usage) {
    demo::showUsage();
    return -1;
  }
  std::string name = demo::getName();
  base::InferenceType inference_type = demo::getInferenceType();
  base::DeviceType device_type = demo::getDeviceType();
  base::ModelType model_type = demo::getModelType();
  bool is_path = demo::isPath();
  std::vector<std::string> model_value = demo::getModelValue();
  std::string input_path = demo::getInputPath();
  std::string ouput_path = demo::getOutputPath();

  // std::string name = YOLO_NAME;
  // base::InferenceType inference_type = base::kInferenceTypeTnn;
  // base::DeviceType device_type = base::kDeviceTypeCodeX86;
  // base::ModelType model_type = base::kModelTypeTnn;
  // bool is_path = true;
  // std::vector<std::string> model_value;
  // model_value.push_back(
  //     "/home/always/huggingface/nndeploy/model_zoo/detect/yolo/"
  //     "yolov6m.tnnproto");
  // model_value.push_back(
  //     "/home/always/huggingface/nndeploy/model_zoo/detect/yolo/"
  //     "yolov6m.tnnmodel");
  // std::string input_path =
  //     "/home/always/huggingface/nndeploy/test_data/detect/sample.jpg";
  // std::string ouput_path =
  //     "/home/always/huggingface/nndeploy/temp/sample_output.jpg";

  model::Packet input("detect_in");
  model::Packet output("detect_out");
  model::Pipeline *pipeline =
      model::createPipeline(name, inference_type, device_type, &input, &output,
                            model_type, is_path, model_value);
  if (pipeline == nullptr) {
    NNDEPLOY_LOGE("pipeline is nullptr");
    return -1;
  }

  NNDEPLOY_TIME_POINT_START("pipeline->init()");
  pipeline->init();
  NNDEPLOY_TIME_POINT_END("pipeline->init()");

  cv::Mat input_mat = cv::imread(input_path);
  input.set(input_mat);
  model::DetectResult result;
  output.set(result);

  NNDEPLOY_TIME_POINT_START("pipeline->run()");
  for (int i = 0; i < 10; ++i) {
    pipeline->run();
  }
  NNDEPLOY_TIME_POINT_END("pipeline->run()");

  draw_box(input_mat, result);
  cv::imwrite(ouput_path, input_mat);

  pipeline->deinit();

  delete pipeline;

  NNDEPLOY_TIME_PROFILER_PRINT();

  NNDEPLOY_TIME_PROFILER_RESET();

  printf("hello world!\n");
  return 0;
}
