/**
 * nndeploy Device Demo:
 * This example demonstrates the basic functionality of nndeploy device
 * management system, focusing on device abstraction, tensor creation and memory
 * management across different devices
 *
 * Main steps:
 * 1. Create and configure device instances (CUDA device with specific device
 * ID)
 * 2. Define tensor descriptors with data type, format and shape specifications
 * 3. Demonstrate tensor creation using different memory management approaches:
 *    - Stack-based tensor creation and manual deletion
 *    - Automatic stack-based tensor with RAII
 *    - Smart pointer-based tensor for automatic memory management
 * 4. Show global tensor creation for cross-function usage
 * 5. Test device-specific tensor operations and memory allocation
 *
 * Key Features Demonstrated:
 * - Device abstraction layer supporting CUDA and other backends
 * - Tensor descriptor system for defining tensor properties
 * - Multiple tensor creation patterns for different use cases
 * - Automatic memory management through RAII and smart pointers
 * - Cross-device tensor operations and data transfer capabilities
 */

#include "flag.h"
#include "nndeploy/base/glic_stl_include.h"
#include "nndeploy/base/shape.h"
#include "nndeploy/base/time_profiler.h"
#include "nndeploy/device/device.h"
#include "nndeploy/device/tensor.h"

using namespace nndeploy;

base::DeviceType g_device_type = base::kDeviceTypeCodeCuda;
device::TensorDesc g_weight_desc(base::dataTypeOf<float>(),
                                 base::kDataFormatOIHW, {32, 1, 3, 3});
device::Tensor g_bias(device::getDevice(g_device_type), g_weight_desc, "bias");

int main(int argc, char const *argv[]) {
  base::DeviceType device_type = base::kDeviceTypeCodeCuda;
  device_type.device_id_ = 0;
  auto device = device::getDevice(device_type);

  device::TensorDesc weight_desc(base::dataTypeOf<float>(),
                                 base::kDataFormatOIHW, {32, 1, 3, 3});
  auto weight = new device::Tensor(device, weight_desc, "weight");
  delete weight;

  device::Tensor bias(device, weight_desc, "bias");

  std::shared_ptr<device::Tensor> s_bias =
      std::make_shared<device::Tensor>(device, weight_desc, "bias");

  return 0;
}