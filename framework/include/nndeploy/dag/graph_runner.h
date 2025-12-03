#ifndef _NNDEPLOY_DAG_GRAPH_RUNNER_H_
#define _NNDEPLOY_DAG_GRAPH_RUNNER_H_

#include <map>
#include <memory>
#include <string>
#include <vector>

#include "nndeploy/base/common.h"
#include "nndeploy/base/status.h"
#include "nndeploy/dag/graph.h"

namespace nndeploy {
namespace dag {

struct GraphRunnerResult {
  base::Status status;
  std::map<std::string, float> time_profiler_map;
  std::map<std::string, std::shared_ptr<RunStatus>> run_status_map;
  std::vector<base::Any> results;
};

/**
 * @brief GraphRunner类，用于运行图计算
 *
 * 这个类提供了构建和运行图的功能，包括性能分析和调试支持
 */
class NNDEPLOY_CC_API GraphRunner {
 public:
  GraphRunner();
  virtual ~GraphRunner();

  /**
   * @brief 运行图计算
   *
   * @param graph_json_str 图的JSON
   * @param name 图的名称
   * @param task_id 任务ID
   * @param time_profiler_map 输出的性能分析结果
   * @param results 输出的计算结果
   * @return base::Status 执行状态
   */
  std::shared_ptr<GraphRunnerResult> run(const std::string& graph_json_str,
                                         const std::string& name = "graph_runner",
                                         const std::string& task_id = "task_id");

  void set_json_file(bool is_json_file);
  void set_dump(bool is_dump);
  void set_time_profile(bool is_time_profile);
  void set_debug(bool is_debug);
  void set_parallel_type(base::ParallelType parallel_type);
  void set_loop_max_flag(bool is_loop_max_flag);
  void set_node_value(const std::string& node_name, const std::string& key,
                      const std::string& value);
  void set_node_value(
      std::map<std::string, std::map<std::string, std::string>> node_value_map);

 protected:
  /**
   * @brief 构建图对象
   *
   * @param graph_json_str 图的JSON字符串
   * @param name 图的名称
   * @return std::shared_ptr<Graph> 构建的图对象
   */
  base::Status buildGraph(const std::string& graph_json_str,
                          const std::string& name);

 private:
  // 禁用拷贝构造和赋值
  GraphRunner(const GraphRunner&) = delete;
  GraphRunner& operator=(const GraphRunner&) = delete;

 private:
  std::shared_ptr<Graph> graph_;
  bool is_json_file_ = true;
  std::string name_;
  std::string task_id_;
  bool is_dump_ = true;
  bool is_time_profile_ = true;
  bool is_debug_ = false;
  base::ParallelType parallel_type_ = base::ParallelType::kParallelTypeNone;
  bool is_loop_max_flag_ = true;
  std::map<std::string, std::map<std::string, std::string>> node_value_map_;
};

}  // namespace dag
}  // namespace nndeploy

#endif  // _NNDEPLOY_DAG_GRAPH_RUNNER_H_
