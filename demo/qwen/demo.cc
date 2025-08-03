#include "flag.h"
#include "nndeploy/base/glic_stl_include.h"
#include "nndeploy/base/time_profiler.h"
#include "nndeploy/codec/codec.h"
#include "nndeploy/dag/node.h"
#include "nndeploy/device/device.h"
#include "nndeploy/framework.h"
#include "nndeploy/qwen/qwen.h"
#include "nndeploy/thread_pool/thread_pool.h"
#include "nndeploy/tokenizer/tokenizer.h"

using namespace nndeploy;
using namespace qwen;

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

  // name of LLM DAG
  std::string name = demo::getName();
  // inference backend
  base::InferenceType inference_type = demo::getInferenceType();
  // device type
  base::DeviceType device_type = demo::getDeviceType();
  // kModelTypeOnnx/kModelTypeMnn/...
  base::ModelType model_type = demo::getModelType();
  bool is_path = demo::isPath();
  // config path
  std::string config_path = demo::getConfigPath();
  // paralle type
  base::ParallelType pt = demo::getParallelType();

  dag::Edge *output = new dag::Edge("llm_out");

  // graph
  dag::Graph *graph = new dag::Graph("demo", {}, {});
  if (graph == nullptr) {
    NNDEPLOY_LOGE("graph is nullptr");
    return -1;
  }

  // config
  QwenConfig config = parseConfig(config_path);

  // prompt node
  dag::Edge *prompt = graph->createEdge("prompt");
  dag::Node *prompt_node =
      graph->createNode<PromptNode>("prompt_node", std::vector<dag::Edge *>{},
                                    std::vector<dag::Edge *>{prompt});

  // prompt params
  PromptParam *prompt_param =
      dynamic_cast<PromptParam *>(prompt_node->getParam());
  prompt_param->prompt_template_ = config.prompt_template_;
  prompt_param->user_content_ = config.prompt_;

  // create dag for llm
  dag::Graph *llm_graph =
      createQwenGraph(name, inference_type, device_type, prompt, output,
                      model_type, is_path, {config_path});
  if (llm_graph == nullptr) {
    NNDEPLOY_LOGE("llm_graph is nullptr");
    return -1;
  }
  graph->addNode(llm_graph);

  // print node
  dag::Node *print_node = graph->createNode<PrintNode>(
      "PrintNode", std::vector<dag::Edge *>{output},
      std::vector<dag::Edge *>{});

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

  graph->dump();

  NNDEPLOY_TIME_POINT_START("graph->run");
  status = graph->run();
  if (status != base::kStatusCodeOk) {
    NNDEPLOY_LOGE("graph deinit failed");
    return -1;
  }
  NNDEPLOY_TIME_POINT_END("graph->run");

  status = graph->deinit();
  if (status != base::kStatusCodeOk) {
    NNDEPLOY_LOGE("graph deinit failed");
    return -1;
  }

  NNDEPLOY_TIME_PROFILER_PRINT("demo");

  delete output;
  delete llm_graph;
  delete graph;

  ret = nndeployFrameworkDeinit();
  if (ret != 0) {
    NNDEPLOY_LOGE("nndeployFrameworkInit failed. ERROR: %d\n", ret);
    return ret;
  }
  return 0;
}