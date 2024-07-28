#include "nndeploy/inference/snpe/snpe_inference_param.h"

namespace nndeploy
{
namespace inference
{

static TypeInferenceParamRegister<TypeInferenceParamCreator<SnpeInferenceParam>>
    g_snpe_inference_param_register(base::kInferenceTypeSnpe);

SnpeInferenceParam::SnpeInferenceParam() : InferenceParam()
{
    model_type_ = base::kModelTypeSnpe;
    device_type_ = device::getDefaultHostDeviceType();
    num_thread_ = 4;
    backup_device_type_ = device::getDefaultHostDeviceType();
}

SnpeInferenceParam::~SnpeInferenceParam() {}

base::Status SnpeInferenceParam::parse(const std::string &json, bool is_path)
{
    std::string json_content = "";
    base::Status status = InferenceParam::parse(json_content, false);
    NNDEPLOY_RETURN_ON_NEQ(status, base::kStatusCodeOk, "parse json failed!");

    return base::kStatusCodeOk;
}

base::Status SnpeInferenceParam::set(const std::string &key, base::Value &value)
{
    base::Status status = base::kStatusCodeOk;

    return base::kStatusCodeOk;
}

base::Status SnpeInferenceParam::get(const std::string &key, base::Value &value)
{
    base::Status status = base::kStatusCodeOK;

    return base::kStatusCodeOk;
}

} // namespace inference
} // namespace nndeploy