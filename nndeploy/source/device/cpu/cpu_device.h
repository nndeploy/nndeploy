
#ifndef _NNDEPLOY_SOURCE_DEVICE_CPU_DEVICE_H_
#define _NNDEPLOY_SOURCE_DEVICE_CPU_DEVICE_H_

#include "nndeploy/source/device/device.h"

namespace nndeploy {
namespace device {

class CpuArchitecture : public Architecture {
 public:
  explicit CpuArchitecture(base::DeviceTypeCode device_type_code);

  virtual ~CpuArchitecture();

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

/**
 * @brief
 *
 */
class NNDEPLOY_CC_API CpuDevice : public Device {
  friend class CpuArchitecture;

 public:
  virtual BufferDesc& toBufferDesc(const MatDesc& desc,
                                   const base::IntVector& config);

  virtual BufferDesc& toBufferDesc(const TensorDesc& desc,
                                   const base::IntVector& config);

  virtual Buffer* allocate(size_t size);
  virtual Buffer* allocate(const BufferDesc& desc);
  virtual void deallocate(Buffer* buffer);

  virtual base::Status copy(Buffer* src, Buffer* dst);
  virtual base::Status download(Buffer* src, Buffer* dst);
  virtual base::Status upload(Buffer* src, Buffer* dst);

 protected:
  CpuDevice(base::DeviceType device_type, void* command_queue = NULL,
            std::string library_path = "")
      : Device(device_type){};
  virtual ~CpuDevice(){};

  virtual base::Status init();
  virtual base::Status deinit();
};

}  // namespace device
}  // namespace nndeploy

#endif
