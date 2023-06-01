#include "nndeploy/source/base/glic_stl_include.h"
#include "nndeploy/source/base/time_measurement.h"
#include "nndeploy/source/inference/inference_param.h"
#include "nntask/source/detect/task.h"




int main(int argc, char *argv[]) {
  nndeploy::base::TimeMeasurement *tm = new nndeploy::base::TimeMeasurement();

  nndeploy::base::DeviceType device_type(nndeploy::base::kDeviceTypeCodeX86, 0);
  nntask::detect::Task *task = new nntask::detect::DETRTask(
      true, nndeploy::base::kInferenceTypeMnn, device_type, "yolov5");
  nndeploy::inference::InferenceParam *inference_param =
      (nndeploy::inference::InferenceParam *)(task->getInferenceParam());
  inference_param->is_path_ = true;
  inference_param->model_value_.push_back("/home/always/Downloads/yolov5s.mnn");//要改
  task->init();
  cv::Mat input_mat = cv::imread("/home/always/Downloads/yolo_input.jpeg");//要改
  nntask::common::Packet input(input_mat);
  nntask::common::DetectParam result;
  nntask::common::Packet output(result);
  task->setInput(input);
  task->setOutput(output);
  task->run();
  draw_box(input_mat, result);
  cv::imwrite("/home/always/Downloads/yolo_result.jpg", input_mat);//要改
  task->deinit();

  printf("The DETR task is done now!\n");
  return 0;
}