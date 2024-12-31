#include "flag.h"
#include "nndeploy/base/glic_stl_include.h"
#include "nndeploy/base/time_profiler.h"
#include "nndeploy/llm/llama2.h"
#include "nndeploy/llm/result.h"
#include "nndeploy/codec/codec.h"
#include "nndeploy/dag/node.h"
#include "nndeploy/device/device.h"
#include "nndeploy/framework.h"
#include "nndeploy/thread_pool/thread_pool.h"

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
  //printHelloWorld() 

  // name of LLM DAG: NNDEPLOY_LLAMA2
  std::string name = demo::getName();
  // 推理后端类型，例如:
  base::InferenceType inference_type = demo::getInferenceType();
  // 推理设备类型，例如:
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

  // 有向无环图graph的输入边packert
  dag::Edge input("llm_in");
  // 有向无环图graph的输出边packert
  dag::Edge output("llm_out");

  // graph
  dag::Graph *graph = new dag::Graph("demo", nullptr, &output);
  if (graph == nullptr) {
    NNDEPLOY_LOGE("graph is nullptr");
    return -1;
  }

  // create DAG for LLM
  dag::Graph *llm_graph =
      dag::createGraph(name, inference_type, device_type, &input, &output,
                       model_type, is_path, model_value);
  if (llm_graph == nullptr) {
    NNDEPLOY_LOGE("llm_graph is nullptr");
    return -1;
  }
  // classification_graph->setTimeProfileFlag(true);
  graph->addNode(llm_graph);

  // infer output 
  dag::Edge *infer_output = graph->createEdge("infer_output");
  dag::Node *infer_token_node = graph->createNode<InferTokenNode>(
      "InferTokenNode", {&input, &output}, {infer_output});


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
  status = llm_graph->dump();

  NNDEPLOY_TIME_POINT_START("graph->run");
  int size = decode_node->getSize();

  NNDEPLOY_LOGE("size = %d.\n", size);
  for (int i = 0; i < size; ++i) {
    status = graph->run();
    if (status != base::kStatusCodeOk) {
      NNDEPLOY_LOGE("graph deinit failed");
      return -1;
    }

    if (pt != base::kParallelTypePipeline) {
      llm::LlmResult*result =
          (llm::LlmResult*)output.getGraphOutputParam();
      if (result == nullptr) {
        NNDEPLOY_LOGE("result is nullptr");
        return -1;
      }
    }
  }

  if (pt == base::kParallelTypePipeline) {
    // NNDEPLOY_LOGE("size = %d.\n", size);
    for (int i = 0; i < size; ++i) {
      llm::LlmResult *result =
          (llm::LlmResult*)output.getGraphOutputParam();
      // NNDEPLOY_LOGE("%p.\n", result);
      if (result == nullptr) {
        NNDEPLOY_LOGE("result is nullptr");
        return -1;
      }
    }
  }

  NNDEPLOY_TIME_POINT_END("graph->run");

  NNDEPLOY_LOGI("hello world!\n");

  // 有向无环图graph反初始化
  status = graph->deinit();
  if (status != base::kStatusCodeOk) {
    NNDEPLOY_LOGE("graph deinit failed");
    return -1;
  }

  NNDEPLOY_TIME_PROFILER_PRINT("demo");

  // 有向无环图graph销毁
  delete encode_node;
  delete decode_node;
  delete llm_graph;
  delete graph;

  NNDEPLOY_LOGI("hello world!\n");

  ret = nndeployFrameworkDeinit();
  if (ret != 0) {
    NNDEPLOY_LOGE("nndeployFrameworkInit failed. ERROR: %d\n", ret);
    return ret;
  }
  return 0;
}