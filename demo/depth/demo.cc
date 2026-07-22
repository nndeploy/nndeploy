/**
 * @file demo.cc
 * @brief Depth Anything 深度估计演示程序
 *
 * 基于 DepthGraph 的深度估计演示。
 *
 * 用法:
 *   ./nndeploy_demo_depth \
 *     --name nndeploy::depth::DepthGraph \
 *     --inference_type kInferenceTypeOnnxRuntime \
 *     --device_type kDeviceTypeCodeX86:0 \
 *     --model_type kModelTypeOnnx \
 *     --is_path \
 *     --model_value /path/to/depth_anything.onnx \
 *     --input_path /path/to/image.jpg \
 *     --output_path /path/to/output.jpg
 */

#include "flag.h"
#include "nndeploy/base/glic_stl_include.h"
#include "nndeploy/base/time_profiler.h"
#include "nndeploy/dag/edge.h"
#include "nndeploy/dag/graph.h"
#include "nndeploy/dag/node.h"
#include "nndeploy/depth/depth_anything/depth_anything.h"
#include "nndeploy/depth/result.h"
#include "nndeploy/device/device.h"

using namespace nndeploy;

/**
 * @brief 将深度估计结果可视化为伪彩色热力图
 */
cv::Mat visualizeDepth(const depth::DepthResult &result) {
  if (result.data_.empty()) {
    return cv::Mat();
  }

  int h = result.height_;
  int w = result.width_;
  float min_val = result.min_val_;
  float max_val = result.max_val_;
  float range = max_val - min_val;
  if (range < 1e-6f) range = 1.0f;

  // 归一化深度值到 [0, 255]
  cv::Mat depth_map(h, w, CV_8UC1);
  for (int y = 0; y < h; ++y) {
    for (int x = 0; x < w; ++x) {
      float val = (result.data_[y * w + x] - min_val) / range;
      depth_map.at<uchar>(y, x) = static_cast<uchar>(val * 255.0f);
    }
  }

  // 应用伪彩色映射 (COLORMAP_INFERNO / COLORMAP_JET)
  cv::Mat color_map;
  cv::applyColorMap(depth_map, color_map, cv::COLORMAP_INFERNO);
  return color_map;
}

int main(int argc, char *argv[]) {
  gflags::ParseCommandLineNonHelpFlags(&argc, &argv, true);
  if (demo::FLAGS_usage) {
    demo::showUsage();
    std::cout << "深度估计演示程序 (Depth Anything)" << std::endl;
    std::cout << "  --name: 模型名称 (默认: nndeploy::depth::DepthGraph)"
              << std::endl;
    std::cout << "  --input_path: 输入图像路径 (必填)" << std::endl;
    std::cout << "  --output_path: 输出深度图路径 (可选)" << std::endl;
    std::cout << "  --model_value: 模型路径 (必填)" << std::endl;
    std::cout << "  --inference_type: 推理后端类型" << std::endl;
    std::cout << "  --device_type: 推理设备类型" << std::endl;
    std::cout << "  --model_type: 模型类型" << std::endl;
    return -1;
  }

  // 解析命令行参数
  std::string name = demo::getName();
  if (name.empty()) {
    name = "nndeploy::depth::DepthGraph";
  }
  base::InferenceType inference_type = demo::getInferenceType();
  base::DeviceType device_type = demo::getDeviceType();
  base::ModelType model_type = demo::getModelType();
  bool is_path = demo::isPath();
  std::vector<std::string> model_value = demo::getModelValue();
  std::string input_path = demo::getInputPath();
  std::string output_path = demo::getOutputPath();

  // 参数校验
  if (input_path.empty()) {
    NNDEPLOY_LOGE("--input_path is required\n");
    return -1;
  }
  if (model_value.empty()) {
    NNDEPLOY_LOGE("--model_value is required\n");
    return -1;
  }

  // 创建 DepthGraph
  depth::DepthGraph graph(name);

  // 设置推理参数
  base::Status status = graph.setInferenceType(inference_type);
  if (status != base::kStatusCodeOk) {
    NNDEPLOY_LOGE("setInferenceType failed: %s\n", status.desc().c_str());
    return -1;
  }
  graph.setInferParam(device_type, model_type, is_path, model_value);

  // 初始化
  status = graph.init();
  if (status != base::kStatusCodeOk) {
    NNDEPLOY_LOGE("graph init failed: %s\n", status.desc().c_str());
    return -1;
  }

  // 加载输入图像
  cv::Mat image = cv::imread(input_path);
  if (image.empty()) {
    NNDEPLOY_LOGE("Failed to load image: %s\n", input_path.c_str());
    graph.deinit();
    return -1;
  }

  // 创建输入边
  dag::Edge input_edge("depth_in");
  input_edge.set(image);

  // 运行推理
  NNDEPLOY_LOGI("Running depth estimation...\n");
  std::vector<dag::Edge *> outputs = graph.forward({&input_edge});

  if (outputs.empty()) {
    NNDEPLOY_LOGE("No output from graph\n");
    graph.deinit();
    return -1;
  }

  // 获取结果
  depth::DepthResult *result = outputs[0]->getGraphOutput<depth::DepthResult>();
  if (result == nullptr) {
    NNDEPLOY_LOGE("Failed to get DepthResult from output\n");
    graph.deinit();
    return -1;
  }

  // 打印结果信息
  NNDEPLOY_LOGI("Depth estimation result:\n");
  NNDEPLOY_LOGI("  size: %dx%d\n", result->width_, result->height_);
  NNDEPLOY_LOGI("  depth range: [%.4f, %.4f]\n", result->min_val_,
                result->max_val_);
  NNDEPLOY_LOGI("  data points: %zu\n", result->data_.size());

  // 保存深度图可视化结果
  if (!output_path.empty()) {
    cv::Mat depth_vis = visualizeDepth(*result);
    if (!depth_vis.empty()) {
      cv::imwrite(output_path, depth_vis);
      NNDEPLOY_LOGI("Depth map saved to: %s\n", output_path.c_str());
    }
  }

  // 清理
  graph.deinit();

  return 0;
}
