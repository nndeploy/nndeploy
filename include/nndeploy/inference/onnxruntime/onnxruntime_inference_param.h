
#ifndef _NNDEPLOY_INFERENCE_ONNXRUNTIME_ONNXRUNTIME_INFERENCE_PARAM_H_
#define _NNDEPLOY_INFERENCE_ONNXRUNTIME_ONNXRUNTIME_INFERENCE_PARAM_H_

#include "nndeploy/inference/inference_param.h"
#include "nndeploy/inference/onnxruntime/onnxruntime_include.h"

namespace nndeploy {
namespace inference {

class OnnxRuntimeInferenceParam : public InferenceParam {
 public:
  OnnxRuntimeInferenceParam();
  virtual ~OnnxRuntimeInferenceParam();

  OnnxRuntimeInferenceParam(const OnnxRuntimeInferenceParam &param) = default;
  OnnxRuntimeInferenceParam &operator=(const OnnxRuntimeInferenceParam &param) =
      default;

  PARAM_COPY(OnnxRuntimeInferenceParam)
  PARAM_COPY_TO(OnnxRuntimeInferenceParam)

  base::Status parse(const std::string &json, bool is_path = true);

  virtual base::Status set(const std::string &key, base::Value &value);

  virtual base::Status get(const std::string &key, base::Value &value);
  /**
   * @brief Level of graph optimization
   * @details
   * -1: mean default(Enable all the optimization strategy)
   * 0: disable all the optimization strategy/1: enable basic strategy
   * 2:enable extend strategy/99: enable all
   */
  int graph_optimization_level_ = 1;
  /**
   * @brief Number of threads to execute the graph
   * @details
   * -1: default. This parameter only will bring effects. while the
   * execution_mode_ set to 1.
   */
  int inter_op_num_threads_ = -1;
  /**
   * @brief Execution mode for the graph
   * @details
   * 0: Sequential mode, execute the operators in graph one by one.
   * 1: Parallel mode, execute the operators in graph parallelly.
   */
  int execution_mode_ = 0;
};

}  // namespace inference
}  // namespace nndeploy

#endif
