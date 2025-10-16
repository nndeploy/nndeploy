#include "flag.h"
#include "nndeploy/base/glic_stl_include.h"
#include "nndeploy/base/time_profiler.h"
#include "nndeploy/codec/codec.h"
#include "nndeploy/dag/node.h"
#include "nndeploy/device/device.h"
#include "nndeploy/llm/llama2.h"
#include "nndeploy/thread_pool/thread_pool.h"
#include "nndeploy/tokenizer/tokenizer.h"

using namespace nndeploy;
using namespace llm;

int main(int argc, char *argv[]) {
  gflags::ParseCommandLineNonHelpFlags(&argc, &argv, true);
  if (demo::FLAGS_usage) {
    demo::showUsage();
    return -1;
  }

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
  // config path
  std::string config_path = demo::getConfigPath();
  // paralle type
  base::ParallelType pt = demo::getParallelType();

  // 有向无环图graph的输出边packert
  dag::Edge *input = new dag::Edge("llm_in");
  dag::Edge *output = new dag::Edge("llm_out");

  // graph
  dag::Graph *graph = new dag::Graph("demo", {}, {output});
  if (graph == nullptr) {
    NNDEPLOY_LOGE("graph is nullptr");
    return -1;
  }

  /* parse config */
  LlmConfig config = parseConfig(config_path);

  /* prompt node */
  dag::Edge *prompt = graph->createEdge("prmompt");
  dag::Node *prompt_node =
      graph->createNode<PromptNode>("prompt_node", {input}, {prompt});

  /* prompt params */
  PromptParam *prompt_param =
      dynamic_cast<PromptParam *>(prompt_node->getParam());
  prompt_param->prompt_template_ = config.prompt_template_;

  prompt_param->user_content_ = config.prompt_;
  // prompt_param->user_content_ = "你好，请问你是谁？";
  // prompt_param->user_content_ = "Hello";
  // prompt_param->user_content_ = "请问今天的天气如何？";

  // create DAG for LLM
  dag::Graph *llama2_graph =
      dag::createGraph(name, inference_type, device_type, prompt, output,
                       model_type, is_path, {config_path});
  if (llama2_graph == nullptr) {
    NNDEPLOY_LOGE("llm_graph is nullptr");
    return -1;
  }
  graph->addNode(llama2_graph);

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

  NNDEPLOY_TIME_POINT_START("graph->run");
  status = graph->run();
  if (status != base::kStatusCodeOk) {
    NNDEPLOY_LOGE("graph deinit failed");
    return -1;
  }

  if (pt != base::kParallelTypePipeline) {
    tokenizer::TokenizerText *result =
        (tokenizer::TokenizerText *)output->getGraphOutputParam();
    if (result == nullptr) {
      NNDEPLOY_LOGE("result is nullptr\n");
      return -1;
    }
    printf("\nQ: %s\n", prompt_param->user_content_.c_str());
    printf("A: %s\n\n", result->texts_[0].c_str());
  }

  if (pt == base::kParallelTypePipeline) {
    tokenizer::TokenizerText *result =
        (tokenizer::TokenizerText *)output->getGraphOutputParam();
    if (result == nullptr) {
      NNDEPLOY_LOGE("result is nullptr");
      return -1;
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

  // 有向无环图graph销毁
  delete input;
  delete output;
  delete llama2_graph;
  delete graph;

  return 0;
}