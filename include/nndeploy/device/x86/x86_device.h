
#ifndef _NNDEPLOY_DEVICE_X86_DEVICE_H_
#define _NNDEPLOY_DEVICE_X86_DEVICE_H_

#include "nndeploy/device/device.h"

namespace nndeploy {
namespace device {

class X86Architecture : public Architecture {
 public:
  explicit X86Architecture(base::DeviceTypeCode device_type_code);

  virtual ~X86Architecture();

  virtual base::Status checkDevice(int device_id = 0,
                                   void* command_queue = nullptr,
                                   std::string library_path = "") override;

  virtual base::Status enableDevice(int device_id = 0,
                                    void* command_queue = nullptr,
                                    std::string library_path = "") override;

  virtual Device* getDevice(int device_id) override;

  virtual std::vector<DeviceInfo> getDeviceInfo(
      std::string library_path = "") override;
};

/**
 * @brief
 *
 */
class NNDEPLOY_CC_API X86Device : public Device {
  friend class X86Architecture;

 public:
  virtual BufferDesc toBufferDesc(const MatDesc& desc,
                                  const base::IntVector& config);

  virtual BufferDesc toBufferDesc(const TensorDesc& desc,
                                  const base::IntVector& config);

  virtual Buffer* allocate(size_t size);
  virtual Buffer* allocate(const BufferDesc& desc);
  virtual void deallocate(Buffer* buffer);

  virtual base::Status copy(Buffer* src, Buffer* dst);
  virtual base::Status download(Buffer* src, Buffer* dst);
  virtual base::Status upload(Buffer* src, Buffer* dst);

 protected:
  X86Device(base::DeviceType device_type, void* command_queue = nullptr,
            std::string library_path = "")
      : Device(device_type){};
  virtual ~X86Device(){};

  virtual base::Status init();
  virtual base::Status deinit();
};

}  // namespace device
}  // namespace nndeploy

#endif