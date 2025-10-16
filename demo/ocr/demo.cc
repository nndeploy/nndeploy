/**
 *
 * This example demonstrates the OCR inference functionality of the nndeploy
 * framework, focusing on OCR inference using C++ API
 */

#include "flag.h"
#include "nndeploy/base/glic_stl_include.h"
#include "nndeploy/base/time_profiler.h"
#include "nndeploy/codec/codec.h"
#include "nndeploy/dag/node.h"
#include "nndeploy/detect/yolo/yolo.h"
#include "nndeploy/detect/yolo/yolox.h"
#include "nndeploy/device/device.h"
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
#include "nndeploy/ocr/classifier.h"
#include "nndeploy/ocr/detector.h"
#include "nndeploy/ocr/drawbox.h"
#include "nndeploy/ocr/ocr.h"
#include "nndeploy/ocr/recognizer.h"
#include "nndeploy/op/op.h"
#include "nndeploy/thread_pool/thread_pool.h"
// #include "nndeploy/op/ascend_cl/op_add.cc"
// #include "nndeploy/op/ascend_cl/ascend_c/op_add_kernel.cc"

using namespace nndeploy;

int main(int argc, char *argv[]) {
  gflags::ParseCommandLineNonHelpFlags(&argc, &argv, true);
  if (demo::FLAGS_usage) {
    demo::showUsage();
    return -1;
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
  std::vector<std::string> classifier_model_value =
      demo::getClassifierModelValue();
  std::vector<std::string> recognizer_model_value =
      demo::getRecognizerModelValue();
  std::vector<std::string> detector_model_value = demo::getDetectorModelValue();
  std::string character_txt_value = demo::getCharacterTxtValue();
  // input path
  std::string input_path = demo::getInputPath();
  // codec flag
  base::CodecFlag codec_flag = demo::getCodecFlag();
  // output path
  std::string ouput_path = demo::getOutputPath();
  // base::kParallelTypePipeline / base::kParallelTypeSequential
  base::ParallelType pt = demo::getParallelType();
  std::vector<std::string> classifier_model_inputs =
      demo::getClassifierModelInputs();
  std::vector<std::string> recognizer_model_inputs =
      demo::getRecognizerModelInputs();
  std::vector<std::string> detector_model_inputs =
      demo::getDetectorModelInputs();
  std::vector<std::string> classifier_model_outputs =
      demo::getClassifierModelOutputs();
  std::vector<std::string> recognizer_model_outputs =
      demo::getRecognizerModelOutputs();
  std::vector<std::string> detector_model_outputs =
      demo::getDetectorModelOutputs();

  // 有向无环图graph的输入边packert
  dag::Edge *input = new dag::Edge("detect_in");
  // 有向无环图graph的输出边packert
  dag::Edge *output = new dag::Edge("detect_out");

  // graph
  dag::Graph *graph = new dag::Graph("demo", {}, {});
  if (graph == nullptr) {
    NNDEPLOY_LOGE("graph is nullptr");
    return -1;
  }

  dag::Edge *detector_output = graph->createEdge("detector_output");
  dag::Edge *detector_input = input;

  // 创建检测模型有向无环图graph
  dag::Graph *detect_graph = nullptr;

  // std::vector<std::string> detect_model_value = {
  //   "/home/general/chunquansang/mydeploy/nndeploy/models/ocr/OCRv5_mobile_det/inference.onnx"
  // };
  detect_graph = new ocr::DetectorGraph(name, {input}, {detector_output});
  auto *v_graph = dynamic_cast<ocr::DetectorGraph *>(detect_graph);
  dag::NodeDesc pre_desc("preprocess", {"detect_in"}, detector_model_inputs);
  dag::NodeDesc infer_desc("infer", detector_model_inputs,
                           detector_model_outputs);
  dag::NodeDesc post_desc("postprocess", detector_model_outputs,
                          {"detector_output"});
  v_graph->make(pre_desc, infer_desc, inference_type, post_desc);
  v_graph->setInferParam(device_type, model_type, is_path,
                         detector_model_value);
  // v_graph->setVersion(version);
  graph->addNode(v_graph, false);

  dag::Edge *draw_output = graph->createEdge("draw_output");
  dag::Node *draw_box_node;
  draw_box_node = graph->createNode<ocr::DrawDetectorBox>(
      "DrawDetectorBox", {input, detector_output}, {draw_output});

  dag::Edge *rotate_crop_output = graph->createEdge("rotate_crop_output");
  // dag::Edge *rotate_crop_input = detector_output; // 最终输出

  dag::Node *rotate_crop_node;
  rotate_crop_node = graph->createNode<ocr::RotateCropImage>(
      "RotateCropImage", {detector_output, input}, {rotate_crop_output});

  dag::Edge *classifier_output =
      graph->createEdge("classifier_output");        // 最终输出
  dag::Edge *classifier_input = rotate_crop_output;  // 接 Detector 的输出

  // std::vector<std::string> classify_model_value = {
  //   "/home/general/chunquansang/mydeploy/nndeploy/models/ocr/ch_ppocr_mobile_v2.0_cls_infer/inference.onnx"
  // };
  // std::vector<std::string> classify_model_inputs = {"x"};
  // std::vector<std::string> classify_model_outputs = {"softmax_0.tmp_0"};
  dag::Graph *classify_graph = nullptr;
  classify_graph = new ocr::ClassifierGraph("Classifier", {classifier_input},
                                            {classifier_output});
  auto *c_graph = dynamic_cast<ocr::ClassifierGraph *>(classify_graph);
  dag::NodeDesc c_pre_desc("c_preprocess", {"rotate_crop_output"},
                           classifier_model_inputs);
  dag::NodeDesc c_infer_desc("c_infer", classifier_model_inputs,
                             classifier_model_outputs);
  dag::NodeDesc c_post_desc("c_postprocess", classifier_model_outputs,
                            {"classifier_output"});
  c_graph->make(c_pre_desc, c_infer_desc, inference_type, c_post_desc);
  c_graph->setInferParam(device_type, model_type, is_path,
                         classifier_model_value);
  // v_graph->setVersion(version);
  graph->addNode(c_graph, false);

  dag::Edge *rotate_180_output = graph->createEdge("rotate_180_output");
  dag::Edge *rotate_180_input = classifier_output;  // 最终输出

  dag::Node *rotate_180_node;
  rotate_180_node = graph->createNode<ocr::RotateImage180>(
      "RotateImage180", {rotate_180_input, rotate_crop_output},
      {rotate_180_output});

  dag::Edge *recognizer_output = output;            // 最终输出
  dag::Edge *recognizer_input = rotate_180_output;  // 接 Detector 的输出

  // std::vector<std::string> recognizer_model_value = {
  //   "/home/general/chunquansang/mydeploy/nndeploy/models/ocr/OCRv5_mobile_rec/inference.onnx"
  //   //
  //   "/home/general/chunquansang/new_nndeploy/nndeploy/tmp/ch_PP-OCRv3_rec_infer.onnx"
  // };
  // std::vector<std::string> recognizer_model_inputs = {"x"};
  // std::vector<std::string> recognizer_model_outputs = {"fetch_name_0"};
  dag::Graph *recognizer_graph = nullptr;
  recognizer_graph = new ocr::RecognizerGraph("Recognizer", {recognizer_input},
                                              {recognizer_output});
  auto *r_graph = dynamic_cast<ocr::RecognizerGraph *>(recognizer_graph);
  dag::NodeDesc r_pre_desc("r_preprocess", {"rotate_180_output"},
                           recognizer_model_inputs);
  dag::NodeDesc r_infer_desc("r_infer", recognizer_model_inputs,
                             recognizer_model_outputs);
  dag::NodeDesc r_post_desc("r_postprocess", recognizer_model_outputs,
                            {"detect_out"});
  r_graph->make(r_pre_desc, r_infer_desc, inference_type, r_post_desc);
  r_graph->setInferParam(device_type, model_type, is_path,
                         recognizer_model_value);
  r_graph->setCharacterPath(character_txt_value);
  // r_graph->setCharacterPath("/home/general/chunquansang/mydeploy/nndeploy/models/ocr/OCRv5_mobile_rec/inference.yml");
  // v_graph->setVersion(version);
  graph->addNode(r_graph, false);

  auto *print_node =
      dynamic_cast<ocr::PrintOcrNode *>(graph->createNode<ocr::PrintOcrNode>(
          "PrintOcrNode", std::vector<dag::Edge *>{recognizer_output},
          std::vector<dag::Edge *>{}));
  print_node->setPath("./112.txt");

  codec::Decode *decode_node = codec::createDecode(
      base::kCodecTypeOpenCV, codec_flag, "decode_node", input);
  graph->addNode(decode_node, false);

  codec::Encode *encode_node = codec::createEncode(
      base::kCodecTypeOpenCV, codec_flag, "encode_node", draw_output);
  graph->addNode(encode_node, false);

  base::Status status = graph->setParallelType(pt);
  if (status != base::kStatusCodeOk) {
    NNDEPLOY_LOGE("graph setParallelType failed");
    return -1;
  }
  graph->setTimeProfileFlag(true);
  NNDEPLOY_TIME_POINT_START("graph->init()");
  status = graph->init();
  if (status != base::kStatusCodeOk) {
    NNDEPLOY_LOGE("graph init failed");
    return -1;
  }
  NNDEPLOY_TIME_POINT_END("graph->init()");
  status = graph->dump();
  NNDEPLOY_TIME_POINT_START("graph->run");
  int size = decode_node->getSize();
  size = 10;
  decode_node->setSize(size);
  decode_node->setPath(input_path);
  std::cout << "----------size---------:" << size << std::endl;
  encode_node->setRefPath(input_path);
  encode_node->setPath(ouput_path);
  for (int i = 0; i < size; ++i) {
    status = graph->run();
    if (status != base::kStatusCodeOk) {
      NNDEPLOY_LOGE("graph deinit failed");
      return -1;
    }

    if (pt != base::kParallelTypePipeline) {
      ocr::OCRResult *result = (ocr::OCRResult *)output->getGraphOutputParam();
      if (result == nullptr) {
        NNDEPLOY_LOGE("result is nullptr");
        return -1;
      }
    }
  }
  if (pt == base::kParallelTypePipeline) {
    NNDEPLOY_LOGE("size = %d.\n", size);
    for (int i = 0; i < size; ++i) {
      ocr::OCRResult *result = (ocr::OCRResult *)output->getGraphOutputParam();
      NNDEPLOY_LOGE("%d %p.\n", i, result);
      if (result == nullptr) {
        NNDEPLOY_LOGE("result is nullptr");
        return -1;
      }
    }
  }
  NNDEPLOY_TIME_POINT_END("graph->run");
  status = dag::saveFile(graph, "ocr_graph.json");
  if (status != base::kStatusCodeOk) {
    NNDEPLOY_LOGE("graph saveFile failed");
    return -1;
  }
  status = graph->deinit();
  if (status != base::kStatusCodeOk) {
    NNDEPLOY_LOGE("graph deinit failed");
    return -1;
  }
  NNDEPLOY_TIME_PROFILER_PRINT("demo");
  NNDEPLOY_TIME_PROFILER_PRINT_REMOVE_WARMUP("demo", 10);
  delete input;
  delete output;

  delete graph;
  return 0;
}
