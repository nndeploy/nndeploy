#ifndef _NNDEPLOY_INFERENCE_SNPE_SNPE_INFERENCE_H_
#define _NNDEPLOY_INFERENCE_SNPE_SNPE_INFERENCE_H_

#include "nndeploy/base/common.h"
#include "nndeploy/base/log.h"
#include "nndeploy/base/macro.h"
#include "nndeploy/base/object.h"
#include "nndeploy/base/shape.h"
#include "nndeploy/base/status.h"
#include "nndeploy/base/value.h"
#include "nndeploy/device/device.h"
#include "nndeploy/device/tensor.h"
#include "nndeploy/inference/inference.h"
#include "nndeploy/inference/inference_param.h"
#include "nndeploy/inference/snpe/snpe_convert.h"
#include "nndeploy/inference/snpe/snpe_include.h"
#include "nndeploy/inference/snpe/snpe_inference_param.h"

namespace nndeploy
{
namespace inference
{

class SnpeInference : public Inference
{
public:
    SnpeInference(base::InferenceType type);
    virtual ~SnpeInference();

    virtual base::Status init();
    virtual base::Status deinit();

    virtual int64_t getMemorySize();

    virtual float getFLOPs();

    virtual device::TensorDesc getInputTensorAlignDesc(const std::string &name);
    virtual device::TensorDesc getOutputTensorAlignDesc(const std::string &name);

    virtual base::Status run();

    virtual device::Tensor *getOutputTensorAfterRun(
        const std::string &name, base::DeviceType device_type, bool is_copy,
        base::DataFormat data_format = base::kDataFormatAuto
    );

private:
    size_t calcSizeFromDims(const zdl::DlSystem::Dimension *dims,
                        size_t rank, size_t elementSize);
    size_t getResizableDim();
    void setResizableDim(size_t resizableDim);

    base::Status allocateInputOutputTensor();
    base::Status deallocateInputOutputTensor();

    typedef unsigned int GLuint;

    // Helper function to fill a single entry of the UserBufferMap with the given user-backed buffer
    void createUserBuffer(zdl::DlSystem::UserBufferMap& userBufferMap,
                        std::unordered_map<std::string, std::vector<uint8_t>>& applicationBuffers,
                        std::vector<std::unique_ptr<zdl::DlSystem::IUserBuffer>>& snpeUserBackedBuffers,
                        std::unique_ptr<zdl::SNPE::SNPE>& snpe,
                        const char * name,
                        const bool isTfNBuffer,
                        int bitWidth);

    void createUserBuffer(zdl::DlSystem::UserBufferMap& userBufferMap,
                        std::unordered_map<std::string, GLuint>& applicationBuffers,
                        std::vector<std::unique_ptr<zdl::DlSystem::IUserBuffer>>& snpeUserBackedBuffers,
                        std::unique_ptr<zdl::SNPE::SNPE>& snpe,
                        const char * name);

    // Create a UserBufferMap of the SNPE network inputs
    void createInputBufferMap(zdl::DlSystem::UserBufferMap& inputMap,
                            std::unordered_map<std::string, std::vector<uint8_t>>& applicationBuffers,
                            std::vector<std::unique_ptr<zdl::DlSystem::IUserBuffer>>& snpeUserBackedBuffers,
                            std::unique_ptr<zdl::SNPE::SNPE>& snpe,
                            const bool isTfNBuffer,
                            int bitWidth);

    void createInputBufferMap(zdl::DlSystem::UserBufferMap& inputMap,
                            std::unordered_map<std::string, GLuint>& applicationBuffers,
                            std::vector<std::unique_ptr<zdl::DlSystem::IUserBuffer>>& snpeUserBackedBuffers,
                            std::unique_ptr<zdl::SNPE::SNPE>& snpe);

    // Create a UserBufferMap of the SNPE network outputs
    void createOutputBufferMap(zdl::DlSystem::UserBufferMap& outputMap,
                            std::unordered_map<std::string, std::vector<uint8_t>>& applicationBuffers,
                            std::vector<std::unique_ptr<zdl::DlSystem::IUserBuffer>>& snpeUserBackedBuffers,
                            std::unique_ptr<zdl::SNPE::SNPE>& snpe,
                            const bool isTfNBuffer,
                            int bitWidth);

    void createITensors(std::unique_ptr<zdl::DlSystem::ITensor>* inputs,
                        std::unique_ptr<zdl::SNPE::SNPE>& snpe,
                        const size_t inputSize);

private:
    size_t resizable_dim;

    base::unique_ptr<zdl::SNPE::SNPE> snpe_;
    base::unique_ptr<zdl::DlSystem::UserBufferMap> input_map_;
    base::unique_ptr<zdl::DlSystem::UserBufferMap> output_map_;
    std::vector<std::unique_ptr<zdl::DlSystem::IUserBuffer>> snpe_user_input_buffers_;
    std::vector<std::unique_ptr<zdl::DlSystem::IUserBuffer>> snpe_user_output_buffers_;
    std::unordered_map<std::string, std::vector<uint8_t>> application_input_buffers_;
    std::unordered_map<std::string, std::vector<uint8_t>> application_output_buffers_;
    std::unique_ptr<zdl::DlSystem::ITensor> inputTensor_;
};

} // namespace inference
} // namespace nndeploy

#endif