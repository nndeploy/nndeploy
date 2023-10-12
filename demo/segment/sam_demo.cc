#include "flag.h"
#include "nndeploy/base/glic_stl_include.h"
#include "nndeploy/base/time_profiler.h"
#include "nndeploy/device/device.h"
#include "nndeploy/model/segment/result.h"
#include "nndeploy/model/segment/segment_anything/sam.h"
#include "nndeploy/model/task.h"

using namespace nndeploy;

int main(int argc, char *argv[]) {
  gflags::ParseCommandLineNonHelpFlags(&argc, &argv, true);
  if (demo::FLAGS_usage) {
    demo::showUsage();
    return -1;
  }

  // 检测模型的有向无环图pipeline名称
  // NNDEPLOY_SAM
  std::string name = demo::getName();
  // 推理后端类型，例如:
  // kInferenceTypeOpenVino / kInferenceTypeTensorRt / kInferenceTypeOnnxRuntime
  base::InferenceType inference_type = demo::getInferenceType();
  // 推理设备类型，例如:
  // kDeviceTypeCodeX86:0/kDeviceTypeCodeCuda:0/...
  base::DeviceType device_type = demo::getDeviceType();
  // 模型类型，例如:
  // kModelTypeOnnx/kModelTypeMnn/...
  base::ModelType model_type = demo::getModelType();
  // 模型是否是路径
  bool is_path = demo::isPath();

  // 模型路径或者模型字符串
  std::vector<std::string> model_value = demo::getModelValue();
  // 有向无环图pipeline的输入边packert
  model::Packet input("segment_in");
  // 有向无环图pipeline的输出边packert
  model::Packet output("segment_out");

  // 创建检测模型有向无环图pipeline
  model::Pipeline *pipeline =
      model::createPipeline(name, inference_type, device_type, &input, &output,
                            model_type, is_path, model_value);
  if (pipeline == nullptr) {
    NNDEPLOY_LOGE("pipeline is nullptr");
    return -1;
  }

  // 初始化有向无环图pipeline
  NNDEPLOY_TIME_POINT_START("pipeline->init()");
  base::Status status = pipeline->init();
  if (status != base::kStatusCodeOk) {
    NNDEPLOY_LOGE("pipeline init failed");
    return -1;
  }
  NNDEPLOY_TIME_POINT_END("pipeline->init()");

  // 有向无环图pipeline的输入图片路径
  std::string input_path = demo::getInputPath();
  // opencv读图
  cv::Mat input_mat = cv::imread(input_path);
  // 将图片写入有向无环图pipeline输入边
  input.set(input_mat);
  // 定义有向无环图pipeline的输出结果
  model::SegmentResult result;
  // 将输出结果写入有向无环图pipeline输出边
  output.set(result);

  status = pipeline->reshape();
  if (status != base::kStatusCodeOk) {
    NNDEPLOY_LOGE("pipeline reshape failed");
    return -1;
  }

  // 有向无环图Pipelinez运行
  NNDEPLOY_TIME_POINT_START("pipeline->run()");
  status = pipeline->run();
  if (status != base::kStatusCodeOk) {
    NNDEPLOY_LOGE("pipeline run failed");
    return -1;
  }
  NNDEPLOY_TIME_POINT_END("pipeline->run()");

  device::Tensor* mask = result.mask_;
  cv::Mat mask_output(mask->getHeight(), mask->getWidth(), CV_32FC1, mask->getPtr());
  cv::threshold(mask_output, mask_output, 0.0, 255.0, cv::THRESH_BINARY);
  mask_output.convertTo(mask_output, CV_8U);
  std::string ouput_path = demo::getOutputPath();
  cv::imwrite(ouput_path, mask_output);

  // 有向无环图pipelinez反初始化
  NNDEPLOY_TIME_POINT_START("pipeline->deinit()");
  status = pipeline->deinit();
  if (status != base::kStatusCodeOk) {
    NNDEPLOY_LOGE("pipeline deinit failed");
    return -1;
  }
  NNDEPLOY_TIME_POINT_END("pipeline->deinit()");

  NNDEPLOY_TIME_PROFILER_PRINT("detetct time profiler");

  // 有向无环图pipelinez销毁
  delete pipeline;

  NNDEPLOY_LOGE("hello world!\n");

  return 0;
}
