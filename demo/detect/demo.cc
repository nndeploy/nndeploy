#include "flag.h"
#include "nndeploy/base/glic_stl_include.h"
#include "nndeploy/base/time_profiler.h"
#include "nndeploy/codec/codec.h"
#include "nndeploy/dag/node.h"
#include "nndeploy/detect/drawbox.h"
#include "nndeploy/detect/yolo/yolo.h"
#include "nndeploy/detect/yolo/yolox.h"
#include "nndeploy/device/device.h"
#include "nndeploy/framework.h"
#include "nndeploy/thread_pool/thread_pool.h"

// + 编译/运行/结果分析
// + 开始->图像解码->前处理->推理->后处理->画框->图像编码->结束
// + 重点看下推理相关组件
//   + infer::Infer (推理节点)
//   + inference::DefaultInference （推理对外的封装接口）
//   + ir::Interpret （模型解释抽象基类）
//   + ir::DefaultInterpret （自定义模型解释模型）
//   + ir::ModelDesc （模型中间表示）
//   + net::Net （计算图）
//   + net::OptPass （图优化）
//   + net::Runtime （运行时）
//   + net::SequentialRuntime （串行运行时）
//   + net::TensorPool （基于图的内存池抽象基类）
//   + net::TensorPool1DOffsetCalculateBySize
//   （基于图的内存池计算，偏移计算方法，基于大小贪心）
//   + op::Op （算子抽象基类）
//   + op::AscendCLOpAdd （封装Ascend Op Library算子实现 + 基于Ascend
//   C算子实现）

// 对应代码
#include "nndeploy/infer/infer.h"
#include "nndeploy/inference/default/default_inference.h"
#include "nndeploy/ir/default_interpret.h"
#include "nndeploy/ir/interpret.h"
#include "nndeploy/ir/ir.h"
#include "nndeploy/net/net.h"
#include "nndeploy/net/optimizer.h"
#include "nndeploy/net/runtime.h"
#include "nndeploy/net/runtime/sequential_runtime.h"
#include "nndeploy/net/tensor_pool.h"
#include "nndeploy/net/tensor_pool/tensor_pool_1d_offset_calculate_by_size.h"
#include "nndeploy/op/op.h"
// #include "nndeploy/op/ascend_cl/op_add.cc"
// #include "nndeploy/op/ascend_cl/ascend_c/op_add_kernel.cc"

#define LOAD_JSON 0

using namespace nndeploy;

DEFINE_int32(yolo_version, 11, "yolo_version");

DEFINE_string(yolo_type, "v", "yolo_type");

int getVersion() { return FLAGS_yolo_version; }

std::string getYoloType() { return FLAGS_yolo_type; }

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
  // kInferenceTypeOpenVino / kInferenceTypeAscendCL / kInferenceTypeTensorRt /
  // kInferenceTypeOnnxRuntime
  base::InferenceType inference_type = demo::getInferenceType();
  // 推理设备类型，例如:
  // kDeviceTypeCodeAscendCL:0/kDeviceTypeCodeX86:0/kDeviceTypeCodeCuda:0/...
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
  // codec flag
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

#if !LOAD_JSON
  // 有向无环图graph的输入边packert
  dag::Edge *input = new dag::Edge("detect_in");
  // 有向无环图graph的输出边packert
  dag::Edge *output = new dag::Edge("detect_out");

  // graph
  dag::Graph *graph = new dag::Graph("demo", {}, {output});
  if (graph == nullptr) {
    NNDEPLOY_LOGE("graph is nullptr");
    return -1;
  }
  // 创建检测模型有向无环图graph
  std::string yolo_type = getYoloType();
  dag::Graph *detect_graph = nullptr;
  if (yolo_type == "v") {
    detect_graph = new detect::YoloGraph(name, {input}, {output});
    auto *v_graph = dynamic_cast<detect::YoloGraph *>(detect_graph);
    dag::NodeDesc pre_desc("preprocess", {"detect_in"}, model_inputs);
    dag::NodeDesc infer_desc("infer", model_inputs, model_outputs);
    dag::NodeDesc post_desc("postprocess", model_outputs, {"detect_out"});
    v_graph->make(pre_desc, infer_desc, inference_type, post_desc);
    v_graph->setInferParam(device_type, model_type, is_path, model_value);
    v_graph->setVersion(version);
    graph->addNode(v_graph);
  } else if (yolo_type == "x") {
    detect_graph = new detect::YoloXGraph(name, {input}, {output});
    auto *x_graph = dynamic_cast<detect::YoloXGraph *>(detect_graph);
    dag::NodeDesc pre_desc("preprocess", {"detect_in"}, model_inputs);
    dag::NodeDesc infer_desc("infer", model_inputs, model_outputs);
    dag::NodeDesc post_desc("postprocess", model_outputs, {"detect_out"});
    x_graph->make(pre_desc, infer_desc, inference_type, post_desc);
    x_graph->setInferParam(device_type, model_type, is_path, model_value);
    graph->addNode(x_graph);
  } else {
    NNDEPLOY_LOGE("yolo_type is not support\n");
    return -1;
  }

  // 解码节点
  codec::DecodeNode *decode_node = codec::createDecodeNode(
      base::kCodecTypeOpenCV, codec_flag, "decode_node", input);
  graph->addNode(decode_node);

  // draw box
  dag::Edge *draw_output = graph->createEdge("draw_output");
  dag::Node *draw_box_node;
  if (name == "nndeploy::detect::YoloMultiConvOutputGraph") {
    draw_box_node = graph->createNode<detect::YoloMultiConvDrawBoxNode>(
        "DrawBoxNode", {input, output}, {draw_output});
  } else {
    draw_box_node = graph->createNode<detect::DrawBoxNode>(
        "DrawBoxNode", {input, output}, {draw_output});
  }

  // 编码节点
  codec::EncodeNode *encode_node = codec::createEncodeNode(
      base::kCodecTypeOpenCV, codec_flag, "encode_node", draw_output);
  graph->addNode(encode_node);
#else
  // dag::Graph *graph = new dag::Graph("demo");
  dag::Graph *graph = dag::loadFile("detect_graph_v5.json");
  if (graph == nullptr) {
    NNDEPLOY_LOGE("graph is nullptr");
    return -1;
  }
  graph->dump();
  detect::YoloGraph *detect_graph =
      (detect::YoloGraph *)graph->getNode("nndeploy::detect::YoloGraph");
  if (detect_graph == nullptr) {
    NNDEPLOY_LOGE("detect_graph is nullptr");
    return -1;
  }
  detect_graph->setInferParam(device_type, model_type, is_path, model_value);
  detect_graph->setVersion(version);
  codec::DecodeNode *decode_node =
      (codec::DecodeNode *)graph->getNode("decode_node");
  codec::EncodeNode *encode_node =
      (codec::EncodeNode *)graph->getNode("encode_node");
  dag::Edge *output = graph->getOutput(0);
#endif

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

  // status = graph->serialize("detect_graph_v5.json");
  // if (status != base::kStatusCodeOk) {
  //   NNDEPLOY_LOGE("graph serialize failed");
  //   return -1;
  // }

  status = graph->dump();

  NNDEPLOY_TIME_POINT_START("graph->run");
  // NNDEPLOY_LOGI("input_path = %s.\n", input_path.c_str());
  // NNDEPLOY_LOGI("ouput_path = %s.\n", ouput_path.c_str());
  int size = decode_node->getSize();
  size = 100;
  decode_node->setSize(size);
  decode_node->setPath(input_path);
  encode_node->setRefPath(input_path);
  encode_node->setPath(ouput_path);
  for (int i = 0; i < size; ++i) {
    status = graph->run();
    if (status != base::kStatusCodeOk) {
      NNDEPLOY_LOGE("graph deinit failed");
      return -1;
    }

    if (pt != base::kParallelTypePipeline) {
      detect::DetectResult *result =
          (detect::DetectResult *)output->getGraphOutputParam();
      if (result == nullptr) {
        NNDEPLOY_LOGE("result is nullptr");
        return -1;
      }
    }
  }

  if (pt == base::kParallelTypePipeline) {
    NNDEPLOY_LOGE("size = %d.\n", size);
    for (int i = 0; i < size; ++i) {
      detect::DetectResult *result =
          (detect::DetectResult *)output->getGraphOutputParam();
      NNDEPLOY_LOGE("%d %p.\n", i, result);
      if (result == nullptr) {
        NNDEPLOY_LOGE("result is nullptr");
        return -1;
      }
    }
  }
  NNDEPLOY_TIME_POINT_END("graph->run");

#if LOAD_JSON
  status = dag::saveFile(graph, "detect_graph_v6.json");
  if (status != base::kStatusCodeOk) {
    NNDEPLOY_LOGE("graph saveFile failed");
    return -1;
  }
#endif

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

#if !LOAD_JSON
  delete encode_node;
  delete decode_node;
  delete detect_graph;
#endif
  delete graph;

  ret = nndeployFrameworkDeinit();
  if (ret != 0) {
    NNDEPLOY_LOGE("nndeployFrameworkInit failed. ERROR: %d\n", ret);
    return ret;
  }

  return 0;
}
