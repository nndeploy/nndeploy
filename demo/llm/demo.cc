#include "flag.h"
#include "nndeploy/base/glic_stl_include.h"
#include "nndeploy/base/time_profiler.h"
#include "nndeploy/codec/codec.h"
#include "nndeploy/dag/graph.h"
#include "nndeploy/dag/node.h"
#include "nndeploy/device/device.h"
#include "nndeploy/llm/llm_infer.h"
#include "nndeploy/thread_pool/thread_pool.h"
#include "nndeploy/tokenizer/tokenizer.h"

using namespace nndeploy;
using namespace llm;

// 直接运行JSON配置文件
int runJson(
    std::string json_file, base::ParallelType pt,
    std::map<std::string, std::map<std::string, std::string>> node_param) {
  std::shared_ptr<dag::Graph> graph = std::make_shared<dag::Graph>("");
  if (!graph) {
    NNDEPLOY_LOGE("create graph failed");
    return -1;
  }
  graph->setNodeValue(node_param);
  base::Status status = graph->loadFile(json_file);
  if (status != base::kStatusCodeOk) {
    NNDEPLOY_LOGE("load file failed. ERROR: %s\n", status.desc().c_str());
    return -1;
  }
  graph->setTimeProfileFlag(true);
  status = graph->init();
  if (status != base::kStatusCodeOk) {
    NNDEPLOY_LOGE("init failed. ERROR: %s\n", status.desc().c_str());
    return -1;
  }
  int count = graph->getLoopCount();
  for (int i = 0; i < count; ++i) {
    status = graph->run();
    if (status != base::kStatusCodeOk) {
      NNDEPLOY_LOGE("run failed. ERROR: %s\n", status.desc().c_str());
      return -1;
    }
    if (pt != base::ParallelType::kParallelTypePipeline) {
      dag::Edge* output = graph->getOutput(0);
      tokenizer::TokenizerText* text =
          output->getGraphOutput<tokenizer::TokenizerText>();
      if (text) {
        // std::cout << text->texts_[0] << std::endl;
        ;
      }
    }
  }
  if (pt == base::ParallelType::kParallelTypePipeline) {
    for (int i = 0; i < count; ++i) {
      dag::Edge* output = graph->getOutput(0);
      tokenizer::TokenizerText* text =
          output->getGraphOutput<tokenizer::TokenizerText>();
      if (text) {
        // std::cout << text->texts_[0] << std::endl;
        ;
      }
    }
  }
  status = graph->deinit();
  if (status != base::kStatusCodeOk) {
    NNDEPLOY_LOGE("deinit failed. ERROR: %s\n", status.desc().c_str());
    return -1;
  }
  NNDEPLOY_TIME_PROFILER_PRINT(graph->getName());
  return 0;
}

// 运行JSON配置文件，移除工作流中的输入和输出节点,用户自己配置输入和输出
int runJsonRemoveInOutNode(
    std::string json_file, base::ParallelType pt,
    std::map<std::string, std::map<std::string, std::string>> node_param) {
  std::shared_ptr<dag::Graph> graph = std::make_shared<dag::Graph>("");
  if (!graph) {
    NNDEPLOY_LOGE("create graph failed");
    return -1;
  }
  graph->setNodeValue(node_param);
  base::Status status = graph->loadFile(json_file);
  if (status != base::kStatusCodeOk) {
    NNDEPLOY_LOGE("load file failed. ERROR: %s\n", status.desc().c_str());
    return -1;
  }
  graph->setTimeProfileFlag(true);
  graph->removeInOutNode();
  status = graph->init();
  if (status != base::kStatusCodeOk) {
    NNDEPLOY_LOGE("init failed. ERROR: %s\n", status.desc().c_str());
    return -1;
  }

  // graph->dump();

  dag::Edge* input = graph->getInput(0);
  dag::Edge* output = graph->getOutput(0);

  int count = graph->getLoopCount();
  for (int i = 0; i < count; ++i) {
    tokenizer::TokenizerText *text = new tokenizer::TokenizerText();
    text->texts_ = { "<|im_start|>user\nWho is Jordan<|im_end|>\n<|im_start|>assistant\n" };
    input->set(text, false);
    status = graph->run();
    if (status != base::kStatusCodeOk) {
      NNDEPLOY_LOGE("run failed. ERROR: %s\n", status.desc().c_str());
      return -1;
    }
    if (pt != base::ParallelType::kParallelTypePipeline) {
      dag::Edge* output = graph->getOutput(0);
      tokenizer::TokenizerText* text =
          output->getGraphOutput<tokenizer::TokenizerText>();
      if (text) {
        // std::cout << "A:" << text->texts_[0] << std::endl;
        ;
      }
    }
  }
  if (pt == base::ParallelType::kParallelTypePipeline) {
    for (int i = 0; i < count; ++i) {
      dag::Edge* output = graph->getOutput(0);
      tokenizer::TokenizerText* text =
          output->getGraphOutput<tokenizer::TokenizerText>();
      if (text) {
        // std::cout << "A:" << text->texts_[0] << std::endl;
        ;
      }
    }
  }
  status = graph->deinit();
  if (status != base::kStatusCodeOk) {
    NNDEPLOY_LOGE("deinit failed. ERROR: %s\n", status.desc().c_str());
    return -1;
  }
  NNDEPLOY_TIME_PROFILER_PRINT(graph->getName());
  return 0;
}

int main(int argc, char* argv[]) {
  gflags::ParseCommandLineNonHelpFlags(&argc, &argv, true);
  if (demo::FLAGS_usage) {
    demo::showUsage();
    return -1;
  }

  bool remove_in_out_node = demo::removeInOutNode();
  base::ParallelType pt = demo::getParallelType();
  std::string json_file = demo::getJsonFile();
  std::map<std::string, std::map<std::string, std::string>> node_param =
      demo::getNodeParam();

  if (remove_in_out_node) {
    return runJsonRemoveInOutNode(json_file, pt, node_param);
  } else {
    return runJson(json_file, pt, node_param);
  }
}