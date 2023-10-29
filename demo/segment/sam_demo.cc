#include "flag.h"
#include "nndeploy/base/glic_stl_include.h"
#include "nndeploy/base/time_profiler.h"
#include "nndeploy/dag/node.h"
#include "nndeploy/device/device.h"
#include "nndeploy/model/segment/result.h"
#include "nndeploy/model/segment/segment_anything/sam.h"

using namespace nndeploy;

int main(int argc, char *argv[]) {
  gflags::ParseCommandLineNonHelpFlags(&argc, &argv, true);
  if (demo::FLAGS_usage) {
    demo::showUsage();
    return -1;
  }

  // 检测模型的有向无环图graph名称
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
  // 有向无环图graph的输入边packert
  dag::Edge input("segment_in");
  // 有向无环图graph的输出边packert
  dag::Edge output("segment_out");

  // 创建检测模型有向无环图graph
  dag::Graph *graph =
      dag::createGraph(name, inference_type, device_type, &input, &output,
                       model_type, is_path, model_value);
  if (graph == nullptr) {
    NNDEPLOY_LOGE("graph is nullptr");
    return -1;
  }

  // 初始化有向无环图graph
  NNDEPLOY_TIME_POINT_START("graph->init()");
  base::Status status = graph->init();
  if (status != base::kStatusCodeOk) {
    NNDEPLOY_LOGE("graph init failed");
    return -1;
  }
  NNDEPLOY_TIME_POINT_END("graph->init()");

  // 有向无环图graph的输入图片路径
  std::string input_path = demo::getInputPath();
  // opencv读图
  cv::Mat input_mat = cv::imread(input_path);
  // 将图片写入有向无环图graph输入边
  input.set(input_mat);
  // 定义有向无环图graph的输出结果
  model::SegmentResult result;
  // 将输出结果写入有向无环图graph输出边
  output.set(result);

  // status = graph->reshape();
  if (status != base::kStatusCodeOk) {
    NNDEPLOY_LOGE("graph reshape failed");
    return -1;
  }

  // 有向无环图Graphz运行
  NNDEPLOY_TIME_POINT_START("graph->run()");
  status = graph->run();
  if (status != base::kStatusCodeOk) {
    NNDEPLOY_LOGE("graph run failed");
    return -1;
  }
  NNDEPLOY_TIME_POINT_END("graph->run()");

  device::Tensor *mask = result.mask_;
  cv::Mat mask_output(mask->getHeight(), mask->getWidth(), CV_32FC1,
                      mask->getPtr());
  cv::threshold(mask_output, mask_output, 0.0, 255.0, cv::THRESH_BINARY);
  mask_output.convertTo(mask_output, CV_8U);
  std::string ouput_path = demo::getOutputPath();
  cv::imwrite(ouput_path, mask_output);

  // 有向无环图graphz反初始化
  NNDEPLOY_TIME_POINT_START("graph->deinit()");
  status = graph->deinit();
  if (status != base::kStatusCodeOk) {
    NNDEPLOY_LOGE("graph deinit failed");
    return -1;
  }
  NNDEPLOY_TIME_POINT_END("graph->deinit()");

  NNDEPLOY_TIME_PROFILER_PRINT("detetct time profiler");

  // 有向无环图graphz销毁
  delete graph;

  NNDEPLOY_LOGE("hello world!\n");

  return 0;
}
