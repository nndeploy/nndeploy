// #ifndef _NNDEPLOY_SOURCE_TASK_PIPELINE_H_
// #define _NNDEPLOY_SOURCE_TASK_PIPELINE_H_

// #include "nndeploy/source/base/basic.h"
// #include "nndeploy/source/base/glic_stl_include.h"
// #include "nndeploy/source/base/log.h"
// #include "nndeploy/source/base/macro.h"
// #include "nndeploy/source/base/object.h"
// #include "nndeploy/source/base/status.h"
// #include "nndeploy/source/base/string.h"
// #include "nndeploy/source/base/value.h"
// #include "nndeploy/source/device/buffer.h"
// #include "nndeploy/source/device/buffer_pool.h"
// #include "nndeploy/source/device/device.h"
// #include "nndeploy/source/device/tensor.h"
// #include "nndeploy/source/inference/inference.h"
// #include "nndeploy/source/inference/inference_param.h"
// #include "nndeploy/source/task/execution.h"
// #include "nndeploy/source/task/packet.h"
// #include "nndeploy/source/task/task.h"

// namespace nndeploy {
// namespace task {

// enum ExecutionType : int32_t {
//   kExecutionTypeStart = 0x0000,
//   kExecutionTypeNormal,
//   kExecutionTypeEnd,
// };

// class PipelineParam : public base::Param {};

// class Pipeline : public Execution {
//  public:
//   Pipeline(const std::string& name = "");
//   virtual ~Pipeline();

//   /**
//    * @brief DAG
//    * # 判断是否有环
//    * # 如何遍历
//    * ## 深度优先遍历
//    * ## 广度优先遍历
//    * # 如何开启多个线程执行
//    * @return base::Status
//    */
//   virtual base::Status init();
//   virtual base::Status deinit();

//   virtual base::Status setInput(Packet& input);
//   virtual base::Status setOutput(Packet& output);

//   virtual base::Status run();

//   virtual base::Status dump(std::ostream& oss = std::cout);

//   virtual base::Status addStart(Execution* execution) {
//     start_execution_repository_.push_back(execution);
//     return base::kStatusCodeOk;
//   }
//   virtual base::Status addExecution(
//       Execution* execution, const std::vector<Execution*>& depend_executions
//       =
//                                 std::initializer_list<Execution*>()) {
//     execution_repository_.insert(std::make_pair(execution,
//     depend_executions)); return base::kStatusCodeOk;
//   }
//   virtual base::Status addEnd(Execution* execution) {
//     end_execution_repository_.push_back(execution);
//     return base::kStatusCodeOk;
//   }

//  private:
//   std::vector<Execution*> start_execution_repository_;
//   std::map<Execution*, std::vector<Execution*>> execution_repository_;
//   std::vector<Execution*> end_execution_repository_;

//   std::vector<Execution*> pipeline_;
// };

// }  // namespace task
// }  // namespace nndeploy

// #endif  // _NNDEPLOY_SOURCE_TASK_PIPELINE_H_
