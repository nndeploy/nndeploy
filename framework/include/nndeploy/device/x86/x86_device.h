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
                                   std::string library_path = "") override;

  virtual base::Status enableDevice(int device_id = 0,
                                    std::string library_path = "") override;

  virtual Device *getDevice(int device_id) override;

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
  virtual BufferDesc toBufferDesc(const TensorDesc &desc,
                                  const base::IntVector &config);

  virtual void *allocate(size_t size);
  virtual void *allocate(const BufferDesc &desc);

  virtual void deallocate(void *ptr);

  virtual base::Status copy(void *src, void *dst, size_t size,
                            Stream *stream = nullptr) override;
  virtual base::Status download(void *src, void *dst, size_t size,
                                Stream *stream = nullptr) override;
  virtual base::Status upload(void *src, void *dst, size_t size,
                              Stream *stream = nullptr) override;

  virtual base::Status copy(Buffer *src, Buffer *dst,
                            Stream *stream = nullptr) override;
  virtual base::Status download(Buffer *src, Buffer *dst,
                                Stream *stream = nullptr) override;
  virtual base::Status upload(Buffer *src, Buffer *dst,
                              Stream *stream = nullptr) override;

 public:
  X86Device(base::DeviceType device_type, std::string library_path = "")
      : Device(device_type) {};
  virtual ~X86Device() {};

  virtual base::Status init();
  virtual base::Status deinit();
};

}  // namespace device
}  // namespace nndeploy

#endif