#include "flag.h"
#include "nndeploy/base/glic_stl_include.h"
#include "nndeploy/base/time_profiler.h"
#include "nndeploy/codec/codec.h"
#include "nndeploy/dag/node.h"
#include "nndeploy/detect/drawbox.h"
#include "nndeploy/detect/yolo/yolo.h"
#include "nndeploy/device/device.h"
#include "nndeploy/framework.h"
#include "nndeploy/thread_pool/thread_pool.h"

using namespace nndeploy;

DEFINE_int32(yolo_version, 11, "yolo_version");

int getVersion() { return FLAGS_yolo_version; }

int main(int argc, char *argv[]) {
  gflags::ParseCommandLineNonHelpFlags(&argc, &argv, true);
  if (demo::FLAGS_usage) {
    demo::showUsage();
    return -1;
  }

  int ret = nndeployFrameworkInit();
  if (ret != 0) {
    NNDEPLOY_LOGE("nndeployFrameworkInit failed. ERROR: %d\n", ret);
    return ret;
  }

  // 检测模型的有向无环图graph名称，例如:
  // NNDEPLOY_YOLOV5/NNDEPLOY_YOLOV6/NNDEPLOY_YOLOV8
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
  // input path
  std::string input_path = demo::getInputPath();
  // input path
  base::CodecFlag codec_flag = demo::getCodecFlag();
  // output path
  std::string ouput_path = demo::getOutputPath();
  // base::kParallelTypePipeline / base::kParallelTypeSequential
  base::ParallelType pt = demo::getParallelType();
  int version = getVersion();
  std::vector<std::string> model_inputs = demo::getModelInputs();
  NNDEPLOY_LOGE("model_inputs = %s.\n", model_inputs[0].c_str());
  std::vector<std::string> model_outputs = demo::getModelOutputs();
  NNDEPLOY_LOGE("model_outputs = %s.\n", model_outputs[0].c_str());

  // 有向无环图graph的输入边packert
  dag::Edge input("detect_in");
  // 有向无环图graph的输出边packert
  dag::Edge output("detect_out");

  // graph
  dag::Graph *graph = new dag::Graph("demo", {}, {&output});
  if (graph == nullptr) {
    NNDEPLOY_LOGE("graph is nullptr");
    return -1;
  }
  // 创建检测模型有向无环图graph
  detect::YoloGraph *detect_graph =
      new detect::YoloGraph(name, {&input}, {&output});
  dag::NodeDesc pre_desc("preprocess", {"detect_in"}, model_inputs);
  dag::NodeDesc infer_desc("infer", model_inputs, model_outputs);
  dag::NodeDesc post_desc("postprocess", model_outputs, {"detect_out"});
  detect_graph->make(pre_desc, infer_desc, inference_type, post_desc);
  detect_graph->setInferParam(device_type, model_type, is_path, model_value);
  detect_graph->setVersion(version);
  // detect_graph->setTimeProfileFlag(true);
  graph->addNode(detect_graph);

  // 解码节点
  codec::DecodeNode *decode_node = codec::createDecodeNode(
      base::kCodecTypeOpenCV, codec_flag, "decode_node", &input);
  graph->addNode(decode_node);

  // draw box
  dag::Edge *draw_output = graph->createEdge("draw_output");
  dag::Node *draw_box_node;
  if (name == "nndeploy::detect::yolo::YoloMultiConvOutputGraph") {
    draw_box_node = graph->createNode<detect::YoloMultiConvDrawBoxNode>(
        "DrawBoxNode", {&input, &output}, {draw_output});
  } else {
    draw_box_node = graph->createNode<detect::DrawBoxNode>(
        "DrawBoxNode", {&input, &output}, {draw_output});
  }

  // 解码节点
  codec::EncodeNode *encode_node = codec::createEncodeNode(
      base::kCodecTypeOpenCV, codec_flag, "encode_node", draw_output);
  graph->addNode(encode_node);

  // 设置pipeline并行
  base::Status status = graph->setParallelType(pt);
  if (status != base::kStatusCodeOk) {
    NNDEPLOY_LOGE("graph setParallelType failed");
    return -1;
  }

  graph->setTimeProfileFlag(true);

  // 初始化有向无环图graph
  NNDEPLOY_TIME_POINT_START("graph->init()");
  status = graph->init();
  if (status != base::kStatusCodeOk) {
    NNDEPLOY_LOGE("graph init failed");
    return -1;
  }
  NNDEPLOY_TIME_POINT_END("graph->init()");

  status = graph->dump();
  status = detect_graph->dump();

  NNDEPLOY_TIME_POINT_START("graph->run");
  decode_node->setPath(input_path);
  encode_node->setRefPath(input_path);
  encode_node->setPath(ouput_path);
  int size = decode_node->getSize();
  size = 100;
  decode_node->setSize(size);
  for (int i = 0; i < size; ++i) {
    status = graph->run();
    if (status != base::kStatusCodeOk) {
      NNDEPLOY_LOGE("graph deinit failed");
      return -1;
    }

    if (pt != base::kParallelTypePipeline) {
      detect::DetectResult *result =
          (detect::DetectResult *)output.getGraphOutputParam();
      if (result == nullptr) {
        NNDEPLOY_LOGE("result is nullptr");
        return -1;
      }
    }
  }

  if (pt == base::kParallelTypePipeline) {
    // NNDEPLOY_LOGE("size = %d.\n", size);
    for (int i = 0; i < size; ++i) {
      detect::DetectResult *result =
          (detect::DetectResult *)output.getGraphOutputParam();
      // NNDEPLOY_LOGE("%p.\n", result);
      if (result == nullptr) {
        NNDEPLOY_LOGE("result is nullptr");
        return -1;
      }
    }
  }
  NNDEPLOY_TIME_POINT_END("graph->run");

  // 有向无环图graph反初始化
  status = graph->deinit();
  if (status != base::kStatusCodeOk) {
    NNDEPLOY_LOGE("graph deinit failed");
    return -1;
  }

  NNDEPLOY_TIME_PROFILER_PRINT("demo");
  // NNDEPLOY_TIME_PROFILER_PRINT_INDEX("demo", 0);
  // NNDEPLOY_TIME_PROFILER_PRINT_INDEX("demo", 1);
  // NNDEPLOY_TIME_PROFILER_PRINT_INDEX("demo", 2);
  // NNDEPLOY_TIME_PROFILER_PRINT_INDEX("demo", 50);
  // NNDEPLOY_TIME_PROFILER_PRINT_INDEX("demo", 99);
  NNDEPLOY_TIME_PROFILER_PRINT_REMOVE_WARMUP("demo", 10);

  // 有向无环图graph销毁
  delete encode_node;
  delete decode_node;
  delete detect_graph;
  delete graph;

  ret = nndeployFrameworkDeinit();
  if (ret != 0) {
    NNDEPLOY_LOGE("nndeployFrameworkInit failed. ERROR: %d\n", ret);
    return ret;
  }

  return 0;
}
