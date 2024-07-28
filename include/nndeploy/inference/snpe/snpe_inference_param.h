#ifndef _NNDEPLOY_INFERENCE_SNPE_SNPE_INFERENCE_PARAM_H_
#define _NNDEPLOY_INFERENCE_SNPE_SNPE_INFERENCE_PARAM_H_

#include "nndeploy/device/device.h"
#include "nndeploy/inference/inference_param.h"
#include "nndeploy/inference/snpe/snpe_include.h"

namespace nndeploy
{
namespace inference
{
class SnpeInferenceParam : public InferenceParam
{
public:
    SnpeInferenceParam();
    virtual ~SnpeInferenceParam();

    SnpeInferenceParam(const SnpeInferenceParam &param) = default;
    SnpeInferenceParam &operator=(const SnpeInferenceParam &param) = default;

    PARAM_COPY(SnpeInferenceParam)
    PARAM_COPY_TO(SnpeInferenceParam)

    base::Status parse(const std::string &json, bool is_path = true);

    virtual base::Status set(const std::string &key, base::Value &value);

    virtual base::Status get(const std::string &key, base::Value &value);

    std::vector<std::string> save_tensors_;
    std::string snpe_runtime_;
    int32_t snpe_perf_mode_;
    int32_t snpe_profiling_level_;
    int32_t snpe_buffer_type_;
    std::vector<std::string> input_names_;
    std::vector<std::string> output_tensor_names_;
    std::vector<std::string> output_layer_names_;

};

} // namespace inference
} // namespace nndeploy

#endif