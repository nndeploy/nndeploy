
#ifndef _NNDEPLOY_SOURCE_DEVICE_CUDA_ARCHITECTURE_H_
#define _NNDEPLOY_SOURCE_DEVICE_CUDA_ARCHITECTURE_H_

#include "nndeploy/source/base/basic.h"
#include "nndeploy/source/base/glic_stl_include.h"
#include "nndeploy/source/base/log.h"
#include "nndeploy/source/base/macro.h"
#include "nndeploy/source/base/object.h"
#include "nndeploy/source/base/status.h"
#include "nndeploy/source/device/architecture.h"
#include "nndeploy/source/device/cuda/cuda_include.h"
#include "nndeploy/source/device/cuda/cuda_util.h"
#include "nndeploy/source/device/device.h"

namespace nndeploy {
namespace device {

class CudaArchitecture : public Architecture {
 public:
  explicit CudaArchitecture(base::DeviceTypeCode device_type_code);

  virtual ~CudaArchitecture();

  virtual base::Status checkDevice(int32_t device_id = 0,
                                   void* command_queue = NULL,
                                   std::string library_path = "") override;

  virtual base::Status enableDevice(int32_t device_id = 0,
                                    void* command_queue = NULL,
                                    std::string library_path = "") override;

  virtual Device* getDevice(int32_t device_id) override;

  virtual std::vector<DeviceInfo> getDeviceInfo(
      std::string library_path = "") override;
};

}  // namespace device
}  // namespace nndeploy

#endif  // _NNDEPLOY_SOURCE_DEVICE_CUDA_ARCHITECTURE_H_