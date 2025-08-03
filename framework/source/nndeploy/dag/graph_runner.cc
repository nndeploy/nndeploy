#include "nndeploy/dag/graph_runner.h"

#include <chrono>
#include <iostream>

#include "nndeploy/base/time_profiler.h"
#include "nndeploy/device/tensor.h"

namespace nndeploy {
namespace dag {

GraphRunner::GraphRunner() {}

GraphRunner::~GraphRunner() {
  if (graph_ && graph_->getInitialized()) {
    graph_->deinit();
    graph_->setInitializedFlag(false);
    graph_.reset();
  }
}

std::shared_ptr<GraphRunnerResult> GraphRunner::run(const std::string& graph_json_str,
                                   const std::string& name,
                                   const std::string& task_id) {
  std::shared_ptr<GraphRunnerResult> result =
      std::make_shared<GraphRunnerResult>();
  result->status = base::kStatusCodeOk;

  name_ = name;
  task_id_ = task_id;

  // 重置时间分析器
  if (is_time_profile_) {
    base::timeProfilerReset();
  }

  // 反序列化图
  if (is_time_profile_) {
    NNDEPLOY_TIME_POINT_START("deserialize_" + name);
  }

  base::Status status = buildGraph(graph_json_str, name);
  if (status != base::kStatusCodeOk) {
    result->status = status;
    return result;
  }

  if (is_time_profile_) {
    NNDEPLOY_TIME_POINT_END("deserialize_" + name);
  }

  // 设置图的配置
  if (graph_) {
    graph_->setTimeProfileFlag(is_time_profile_);
    graph_->setDebugFlag(is_debug_);
    if (parallel_type_ != base::ParallelType::kParallelTypeNone) {
      graph_->setParallelType(parallel_type_);
    }
    graph_->setLoopMaxFlag(is_loop_max_flag_);
  }

  // 初始化图
  if (is_time_profile_) {
    base::timePointStart("init_" + name);
  }

  status = graph_->init();
  if (status != base::kStatusCodeOk) {
    result->status = status;
    return result;
  }

  if (is_time_profile_) {
    base::timePointEnd("init_" + name);
  }

  // 获取并行类型
  base::ParallelType parallel_type = graph_->getParallelType();

  // 转储图信息
  if (is_dump_) {
    graph_->dump();
  }

  // 运行图
  if (is_time_profile_) {
    base::timePointStart("sum_" + name);
  }

  int count = graph_->getLoopCount();
  for (int i = 0; i < count; ++i) {
    auto start_time = std::chrono::high_resolution_clock::now();

    status = graph_->run();
    if (status != base::kStatusCodeOk) {
      result->status = status;
      return result;
    }

    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(
        end_time - start_time);
    std::cout << "run " << i << " times, time: " << duration.count() / 1000.0
              << " ms" << std::endl;

    // 对于非流水线模式，立即获取输出
    if (parallel_type != base::ParallelType::kParallelTypePipeline) {
      std::vector<Edge*> outputs = graph_->getAllOutput();
    }
  }

  // 对于流水线模式，在所有运行完成后获取输出
  if (parallel_type == base::ParallelType::kParallelTypePipeline) {
    for (int i = 0; i < count; ++i) {
      std::vector<Edge*> outputs = graph_->getAllOutput();
    }
  }

  // 同步
  bool sync_flag = graph_->synchronize();
  if (!sync_flag) {
    result->status = base::kStatusCodeErrorDag;
    return result;
  }

  if (is_time_profile_) {
    base::timePointEnd("sum_" + name);
  }

  // 获取性能分析结果
  if (is_time_profile_) {
    result->time_profiler_map["sum_" + name] =
        base::timeProfilerGetAverageTime("sum_" + name);
    result->time_profiler_map["init_" + name] =
        base::timeProfilerGetAverageTime("init_" + name);
    result->time_profiler_map["deserialize_" + name] =
        base::timeProfilerGetAverageTime("deserialize_" + name);

    // 打印性能分析结果
    base::timeProfilerPrint(name);
  }

  // 获取运行状态
  result->run_status_map = graph_->getNodesRunStatusRecursive();
  for (const auto& pair : result->run_status_map) {
    std::cout << pair.first << ": " << pair.second->getStatus() << std::endl;
  }

  // 反初始化图
  // graph_->deinit();

  return result;
}

void GraphRunner::set_json_file(bool is_json_file) {
  is_json_file_ = is_json_file;
}

void GraphRunner::set_dump(bool is_dump) { is_dump_ = is_dump; }

void GraphRunner::set_time_profile(bool is_time_profile) {
  is_time_profile_ = is_time_profile;
}

void GraphRunner::set_debug(bool is_debug) { is_debug_ = is_debug; }

void GraphRunner::set_parallel_type(base::ParallelType parallel_type) {
  parallel_type_ = parallel_type;
}

void GraphRunner::set_loop_max_flag(bool is_loop_max_flag) {
  is_loop_max_flag_ = is_loop_max_flag;
}

base::Status GraphRunner::buildGraph(const std::string& graph_json_str,
                                     const std::string& name) {
  graph_ = std::make_shared<Graph>(name);
  if (!graph_) {
    return base::kStatusCodeErrorOutOfMemory;
  }

  base::Status status;
  if (is_json_file_) {
    // 从文件加载
    status = graph_->loadFile(graph_json_str);
  } else {
    // 从字符串反序列化
    status = graph_->deserialize(graph_json_str);
  }

  if (status != base::kStatusCodeOk) {
    graph_.reset();
    return status;
  }

  return base::kStatusCodeOk;
}

}  // namespace dag
}  // namespace nndeploy
