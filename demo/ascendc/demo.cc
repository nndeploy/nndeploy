#include <cassert>
#include <chrono>
#include <fstream>
#include <iostream>
#include <string>

#include "nndeploy/base/glic_stl_include.h"
#include "nndeploy/device/ascend_cl/ascend_cl_util.h"
#include "nndeploy/device/device.h"
#include "nndeploy/device/tensor.h"
#include "nndeploy/framework.h"
#include "nndeploy/op/op.h"
#include "nndeploy/op/op_add.h"

using namespace nndeploy;

int main(int argc, char **argv) {
  int ret = nndeployFrameworkInit();
  if (ret != 0) {
    NNDEPLOY_LOGE("nndeployFrameworkInit failed. ERROR: %d\n", ret);
    return ret;
  }

  base::DeviceType cpu_device_type;
  cpu_device_type.code_ = base::kDeviceTypeCodeCpu;
  device::Device *cpu_device = device::getDevice(cpu_device_type);

  base::DeviceType ascendc_device_type;
  ascendc_device_type.code_ = base::kDeviceTypeCodeAscendCL;
  ascendc_device_type.device_id_ = 1;
  device::Device *ascendc_device = device::getDevice(ascendc_device_type);

  device::TensorDesc desc;
  desc.data_type_ = base::dataTypeOf<half_float::half>();
  desc.data_format_ = base::kDataFormatNC;
  desc.shape_ = {32, 32, 64, 64};

  const int total_size = 32 * 32 * 64 * 64 * sizeof(int16_t);
  uint8_t *a_data = (uint8_t *)malloc(total_size);
  uint8_t *b_data = (uint8_t *)malloc(total_size);
  uint8_t *c_data = (uint8_t *)malloc(total_size);

  for (int i = 0; i < total_size / sizeof(int16_t); i++) {
    a_data[i] = 1;
    b_data[i] = 1;
  }

  device::Tensor *a_host = new device::Tensor(cpu_device, desc, (void *)a_data);
  device::Tensor *b_host = new device::Tensor(cpu_device, desc, (void *)b_data);
  device::Tensor *c_host = new device::Tensor(cpu_device, desc, (void *)c_data);

  device::Tensor *a_device = new device::Tensor(ascendc_device, desc);
  device::Tensor *b_device = new device::Tensor(ascendc_device, desc);
  device::Tensor *c_device = new device::Tensor(ascendc_device, desc);

  b_host->copyTo(b_device);
  a_host->copyTo(a_device);

  base::Status status = op::add(a_device, b_device, c_device);
  if (status != base::kStatusCodeOk) {
    NNDEPLOY_LOGE("op::mul failed. ERROR: %d\n", status.getStatusCode());
    return status.getStatusCode();
  }

  c_device->copyTo(c_host);

  for (int i = 0; i < total_size / sizeof(int16_t); i++) {
    if (c_data[i] != 2) {
      NNDEPLOY_LOGE("op::mul failed. ERROR: %d\n", c_data[i]);
      return -1;
    }
  }

  std::cout << "run ascendc_op success" << std::endl;

  free(a_data);
  free(b_data);
  free(c_data);

  ret = nndeployFrameworkDeinit();
  if (ret != 0) {
    NNDEPLOY_LOGE("nndeployFrameworkInit failed. ERROR: %d\n", ret);
    return ret;
  }

  return 0;
}