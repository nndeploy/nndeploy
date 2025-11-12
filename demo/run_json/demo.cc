/**
 * nndeploy Run Json Demo:
 * Generic tool to execute JSON configuration files
 */

#include <time.h>

#include <fstream>
#include <iostream>
#include <memory>
#include <string>

#include "flag.h"
#include "nndeploy/base/dlopen.h"
#include "nndeploy/dag/graph_runner.h"



#include "nndeploy/base/macro.h"

using namespace nndeploy;

// DEFINE_string(json_file, "", "json_file");
// DEFINE_string(task_id, "", "task_id");

int main(int argc, char* argv[]) {
  gflags::ParseCommandLineNonHelpFlags(&argc, &argv, true);
  if (demo::FLAGS_usage) {
    demo::showUsage();
    return -1;
  }
  std::string json_file = demo::getJsonFile();
  std::string name = demo::getName();
  std::string task_id = demo::getTaskId();
  bool success = demo::loadPlugin();
  if (!success) {
    NNDEPLOY_LOGE("load plugin failed");
    return -1;
  }
  std::map<std::string, std::map<std::string, std::string>> node_param =
      demo::getNodeParam();
  bool dump = demo::dump();
  bool debug = demo::debug();
  bool time_profile = demo::timeProfile();

  dag::GraphRunner* graph_runner = new dag::GraphRunner();
  if (graph_runner == nullptr) {
    NNDEPLOY_LOGE("create graph runner failed");
    return -1;
  }
  graph_runner->set_dump(dump);
  graph_runner->set_debug(debug);
  graph_runner->set_time_profile(time_profile);
  graph_runner->set_parallel_type(base::ParallelType::kParallelTypeNone);
  graph_runner->set_node_value(node_param);

  auto result = graph_runner->run(json_file, name, task_id);
  if (result->status != base::kStatusCodeOk) {
    NNDEPLOY_LOGE("run failed. ERROR: %s\n", result->status.desc().c_str());
    return -1;
  }

  delete graph_runner;

  return 0;
}
