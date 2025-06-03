#include "flag.h"
#include "nndeploy/base/glic_stl_include.h"
#include "nndeploy/base/time_profiler.h"
#include "nndeploy/codec/codec.h"
#include "nndeploy/dag/node.h"
#include "nndeploy/device/device.h"
#include "nndeploy/framework.h"
#include "nndeploy/thread_pool/thread_pool.h"
#include "nndeploy/track/fairmot/fairmot.h"
#include "nndeploy/track/result.h"
#include "nndeploy/track/vis_mot.h"

using namespace nndeploy;

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

  std::string name = demo::getName();
  base::InferenceType inference_type = demo::getInferenceType();
  base::DeviceType device_type = demo::getDeviceType();
  base::ModelType model_type = demo::getModelType();
  bool is_path = demo::isPath();
  std::vector<std::string> model_value = demo::getModelValue();
  std::string input_path = demo::getInputPath();
  std::string output_path = demo::getOutputPath();
  base::CodecFlag codec_flag = demo::getCodecFlag();
  base::ParallelType pt = demo::getParallelType();
  std::vector<std::string> model_inputs = demo::getModelInputs();
  std::vector<std::string> model_outputs = demo::getModelOutputs();

  dag::Edge *input = new dag::Edge("track_in");
  dag::Edge *output = new dag::Edge("track_out");

  dag::Graph *graph = new dag::Graph("demo", {}, {output});
  if (graph == nullptr) {
    NNDEPLOY_LOGE("graph is nullptr");
    return -1;
  }

  track::FairMotGraph *track_graph =
      new track::FairMotGraph(name, {input}, {output});
  dag::NodeDesc pre_desc("preprocess", {"track_in"}, model_inputs);
  dag::NodeDesc infer_desc("infer", model_inputs, model_outputs);
  dag::NodeDesc post_desc("postprocess", model_outputs, {"track_out"});
  track_graph->make(pre_desc, infer_desc, inference_type, post_desc);
  track_graph->setInferParam(device_type, model_type, is_path, model_value);
  graph->addNode(track_graph);

  // Video decoder
  codec::DecodeNode *decode_node = codec::createDecodeNode(
      base::kCodecTypeOpenCV, codec_flag, "decode_node", input);
  graph->addNode(decode_node);

  // draw box
  dag::Edge *vismot_img = graph->createEdge("vismot_img");
  dag::Node *vismot_node;
  vismot_node = graph->createNode<track::VisMOTNode>(
      "vismot_node", {input, output}, {vismot_img});

  // Video encoder
  codec::EncodeNode *encode_node = codec::createEncodeNode(
      base::kCodecTypeOpenCV, codec_flag, "encode_node", vismot_img);
  graph->addNode(encode_node);

  // Set pipeline
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

  NNDEPLOY_TIME_POINT_START("graph->run");
  decode_node->setPath(input_path);
  encode_node->setRefPath(input_path);
  encode_node->setPath(output_path);
  int size = decode_node->getSize();
  for (int i = 0; i < size; ++i) {
    status = graph->run();
    if (status != base::kStatusCodeOk) {
      NNDEPLOY_LOGE("graph deinit failed");
      return -1;
    }

    if (pt != base::kParallelTypePipeline) {
      track::MOTResult *result =
          (track::MOTResult *)output->getGraphOutputParam();
      if (result == nullptr) {
        NNDEPLOY_LOGE("result is nullptr");
        return -1;
      }
    }
  }

  if (pt == base::kParallelTypePipeline) {
    NNDEPLOY_LOGE("size = %d.\n", size);
    for (int i = 0; i < size; ++i) {
      track::MOTResult *result =
          (track::MOTResult *)output->getGraphOutputParam();
      NNDEPLOY_LOGE("%d %p.\n", i, result);
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

  delete encode_node;
  delete decode_node;
  delete track_graph;
  delete graph;

  ret = nndeployFrameworkDeinit();
  if (ret != 0) {
    NNDEPLOY_LOGE("nndeployFrameworkInit failed. ERROR: %d\n", ret);
    return ret;
  }

  return 0;
}