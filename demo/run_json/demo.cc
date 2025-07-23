#include <fstream>
#include <iostream>
#include <memory>
#include <string>
#include <time.h>

#include "flag.h"
#include "nndeploy/dag/graph_runner.h"
#include "nndeploy/framework.h"

using namespace nndeploy;

DEFINE_string(json_file, "", "json_file");
DEFINE_string(task_id, "", "task_id");

int main(int argc, char* argv[]) {
  int ret = nndeployFrameworkInit();
  if (ret != 0) {
    NNDEPLOY_LOGE("nndeployFrameworkInit failed. ERROR: %d\n", ret);
    return ret;
  }

  gflags::ParseCommandLineNonHelpFlags(&argc, &argv, true);
  if (demo::FLAGS_usage) {
    demo::showUsage();
    return -1;
  }
  std::string json_file = FLAGS_json_file;
  std::string name = demo::getName();
  std::string task_id = FLAGS_task_id;

  dag::GraphRunner* graph_runner = new dag::GraphRunner();
  auto result = graph_runner->run(json_file, name, task_id);
  if (result->status != base::kStatusCodeOk) {
    NNDEPLOY_LOGE("run failed. ERROR: %s\n", result->status.desc().c_str());
    return -1;
  }

  delete graph_runner;

  ret = nndeployFrameworkDeinit();
  if (ret != 0) {
    NNDEPLOY_LOGE("nndeployFrameworkDeinit failed. ERROR: %d\n", ret);
    return ret;
  }

  return 0;
}
