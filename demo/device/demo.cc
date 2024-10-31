#include "flag.h"
#include "nndeploy/base/glic_stl_include.h"
#include "nndeploy/base/shape.h"
#include "nndeploy/base/time_profiler.h"
#include "nndeploy/device/device.h"
#include "nndeploy/device/tensor.h"

using namespace nndeploy;

int main(int argc, char const *argv[]) {
  base::DeviceType device_type = base::kDeviceTypeCodeCuda;
  device_type.device_id_ = 0;
  auto device = device::getDevice(device_type);

  device::TensorDesc weight_desc(base::dataTypeOf<float>(),
                                 base::kDataFormatOIHW, {32, 1, 3, 3});
  auto weight = new device::Tensor(device, weight_desc, "weight");
  delete weight;

  NNDEPLOY_LOGE("hello world\n");

  device::Tensor bias(device, weight_desc, "bias");

  NNDEPLOY_LOGE("hello world\n");

  device::disableDevice();

  NNDEPLOY_LOGE("hello world\n");
  return 0;
}