
#include "nndeploy/source/task/pipeline.h"

namespace nndeploy {
namespace task {

Pipeline::Pipeline(const std::string& name) : Execution(name) {
  param_ = (std::make_shared<PipelineParam>());
};
Pipeline::~Pipeline(){};

/**
 * @brief DAG
 * # 判断是否有环
 * # 如何遍历
 * ## 深度优先遍历
 * ## 广度优先遍历
 * # 如何开启多个线程执行
 * @return base::Status
 */
base::Status Pipeline::init() {
  for (auto& execution : pipeline_) {
    NNDEPLOY_RETURN_ON_NEQ(execution->init(), base::kStatusCodeOk);
  }
}
base::Status Pipeline::deinit() {
  for (auto& execution : pipeline_) {
    NNDEPLOY_RETURN_ON_NEQ(execution->deinit(), base::kStatusCodeOk);
  }
  return base::kStatusCodeOk;
}

base::Status Pipeline::setInput(Packet& input) {
  for (auto& execution : start_execution_repository_) {
    NNDEPLOY_RETURN_ON_NEQ(execution->setInput(input), base::kStatusCodeOk);
  }
  return base::kStatusCodeOk;
}
base::Status Pipeline::setOutput(Packet& output) {
  for (auto& execution : end_execution_repository_) {
    NNDEPLOY_RETURN_ON_NEQ(execution->setOutput(output), base::kStatusCodeOk);
  }
  return base::kStatusCodeOk;
}

base::Status Pipeline::run() {
  for (auto& execution : pipeline_) {
    NNDEPLOY_RETURN_ON_NEQ(execution->run(), base::kStatusCodeOk);
  }
  return base::kStatusCodeOk;
}

base::Status Pipeline::dump(std::ostream& oss = std::cout) {
  return base::kStatusCodeOk;
}

base::Status Pipeline::addStart(Execution* execution) {
  start_execution_repository_.push_back(execution);
  return base::kStatusCodeOk;
}
base::Status Pipeline::addExecution(
    Execution* execution, const std::vector<Execution*>& depend_executions =
                              std::initializer_list<Execution*>()) {
  execution_repository_.insert(std::make_pair(execution, depend_executions));
  return base::kStatusCodeOk;
}
base::Status Pipeline::addEnd(Execution* execution) {
  end_execution_repository_.push_back(execution);
  return base::kStatusCodeOk;
}

}  // namespace task
}  // namespace nndeploy
