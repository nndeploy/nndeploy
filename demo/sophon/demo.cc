#include <iostream>
#include <vector>

#include "nndeploy/base/status.h"
#include "nndeploy/base/log.h"
#include "nndeploy/device/tensor.h"
#include "nndeploy/inference/inference.h"
#include "nndeploy/inference/inference_param.h"
#include "nndeploy/inference/sophon/sophon_inference.h"
#include "nndeploy/inference/sophon/sophon_inference_param.h"

using namespace nndeploy;

int main(int argc, char *argv[]) {
  if (argc < 2) {
    NNDEPLOY_LOGE("Usage: %s <bmodel_path> [tpu_id]\n", argv[0]);
    NNDEPLOY_LOGE("Example: %s ./model.bmodel 0\n", argv[0]);
    return -1;
  }

  std::string model_path = argv[1];
  int tpu_id = 0;
  if (argc >= 3) {
    tpu_id = std::atoi(argv[2]);
  }

  // Step 1: Create inference parameter
  inference::SophonInferenceParam *param =
      new inference::SophonInferenceParam(base::kInferenceTypeSophon);
  param->is_path_ = true;
  param->model_value_.push_back(model_path);
  param->device_type_ = base::DeviceType(base::kDeviceTypeCodeSophonNpu, tpu_id);
  param->tpu_id_ = tpu_id;
  param->io_mode_ = sail::SYSIO;
  param->num_thread_ = 4;

  // Step 2: Create and initialize inference
  inference::Inference *infer =
      new inference::SophonInference(base::kInferenceTypeSophon);
  infer->setInferenceParam(
      std::shared_ptr<inference::InferenceParam>(param));

  base::Status status = infer->init();
  if (status != base::kStatusCodeOk) {
    NNDEPLOY_LOGE("Failed to init Sophon inference.\n");
    delete infer;
    return -1;
  }
  NNDEPLOY_LOGE("Sophon inference initialized successfully.\n");

  // Step 3: Get input tensor info and set input data
  std::vector<std::string> input_names = infer->getInputNames();
  NNDEPLOY_LOGE("Number of inputs: %zu\n", input_names.size());

  for (const auto &name : input_names) {
    device::Tensor *input_tensor = infer->getInputTensor(name);

    if (input_tensor == nullptr) {
      NNDEPLOY_LOGE("Failed to get input tensor: %s\n", name.c_str());
      infer->deinit();
      delete infer;
      return -1;
    }

    device::TensorDesc desc = input_tensor->getDesc();
    NNDEPLOY_LOGE("Input [%s] shape: ", name.c_str());
    for (auto dim : desc.shape_) {
      NNDEPLOY_LOGE("%d ", dim);
    }
    NNDEPLOY_LOGE("\n");

    // Fill input with zeros (replace with actual data for real use)
    size_t data_size = input_tensor->getBuffer()->getSize();
    memset(input_tensor->getBuffer()->getData(), 0, data_size);
  }

  // Step 4: Run inference
  NNDEPLOY_LOGE("Running inference...\n");
  status = infer->run();
  if (status != base::kStatusCodeOk) {
    NNDEPLOY_LOGE("Inference failed.\n");
    infer->deinit();
    delete infer;
    return -1;
  }
  NNDEPLOY_LOGE("Inference completed successfully.\n");

  // Step 5: Get and display output
  std::vector<std::string> output_names = infer->getOutputNames();
  NNDEPLOY_LOGE("Number of outputs: %zu\n", output_names.size());

  for (const auto &name : output_names) {
    device::Tensor *output_tensor = infer->getOutputTensor(name);
    if (output_tensor == nullptr) {
      NNDEPLOY_LOGE("Failed to get output tensor: %s\n", name.c_str());
      continue;
    }

    device::TensorDesc desc = output_tensor->getDesc();
    NNDEPLOY_LOGE("Output [%s] shape: ", name.c_str());
    for (auto dim : desc.shape_) {
      NNDEPLOY_LOGE("%d ", dim);
    }
    NNDEPLOY_LOGE("\n");

    // Print first few values of output
    float *data = static_cast<float *>(output_tensor->getBuffer()->getData());
    size_t num_elements = output_tensor->getNumElements();
    size_t print_count = std::min<size_t>(num_elements, 10);
    NNDEPLOY_LOGE("Output [%s] first %zu values: ", name.c_str(), print_count);
    for (size_t i = 0; i < print_count; ++i) {
      NNDEPLOY_LOGE("%f ", data[i]);
    }
    NNDEPLOY_LOGE("\n");
  }

  // Step 6: Get output tensor after run (with copy)
  for (const auto &name : output_names) {
    device::DeviceType host_device(base::kDeviceTypeCodeCpu, 0);
    device::Tensor *output = infer->getOutputTensorAfterRun(
        name, host_device, true, base::kDataFormatAuto);
    if (output != nullptr) {
      NNDEPLOY_LOGE("Got output [%s] on host device.\n", name.c_str());
      delete output;
    }
  }

  // Step 7: Cleanup
  infer->deinit();
  delete infer;
  NNDEPLOY_LOGE("Sophon inference demo completed.\n");

  return 0;
}
