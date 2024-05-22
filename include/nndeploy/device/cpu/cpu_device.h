
#ifndef _NNDEPLOY_DEVICE_CPU_DEVICE_H_
#define _NNDEPLOY_DEVICE_CPU_DEVICE_H_

#include "nndeploy/device/device.h"

namespace nndeploy {
namespace device {

class CpuArchitecture : public Architecture {
 public:
  explicit CpuArchitecture(base::DeviceTypeCode device_type_code);

  virtual ~CpuArchitecture();

  virtual base::Status checkDevice(int device_id, void *command_queue = nullptr,
                                   std::string library_path = "") override;

  virtual base::Status enableDevice(int device_id,
                                    void *command_queue = nullptr,
                                    std::string library_path = "") override;

  virtual Device *getDevice(int device_id) override;

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
  virtual BufferDesc toBufferDesc(const TensorDesc &desc,
                                  const base::IntVector &config);

  virtual void *allocate(size_t size);
  virtual void *allocate(const BufferDesc &desc);

  virtual void deallocate(void *ptr);

  virtual base::Status copy(void *src, void *dst, size_t size);
  virtual base::Status download(void *src, void *dst, size_t size);
  virtual base::Status upload(void *src, void *dst, size_t size);

  virtual base::Status copy(Buffer *src, Buffer *dst);
  virtual base::Status download(Buffer *src, Buffer *dst);
  virtual base::Status upload(Buffer *src, Buffer *dst);

 protected:
  CpuDevice(base::DeviceType device_type, void *command_queue = nullptr,
            std::string library_path = "")
      : Device(device_type){};
  virtual ~CpuDevice(){};

  virtual base::Status init();
  virtual base::Status deinit();
};

}  // namespace device
}  // namespace nndeploy

#endif
