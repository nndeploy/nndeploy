

#ifndef _NNDEPLOY_DEVICE_ASCEND_CL_DEVICE_H_
#define _NNDEPLOY_DEVICE_ASCEND_CL_DEVICE_H_

#include "nndeploy/device/ascend_cl/ascend_cl_include.h"
#include "nndeploy/device/device.h"

namespace nndeploy {
namespace device {

class AscendCLArchitecture : public Architecture {
 public:
  explicit AscendCLArchitecture(base::DeviceTypeCode device_type_code);

  virtual ~AscendCLArchitecture();

  virtual base::Status checkDevice(int device_id = 0,
                                   void *command_queue = nullptr,
                                   std::string library_path = "") override;

  virtual base::Status enableDevice(int device_id = 0,
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
class NNDEPLOY_CC_API AscendCLDevice : public Device {
  friend class AscendCLArchitecture;

 public:
  virtual BufferDesc toBufferDesc(const TensorDesc &desc,
                                  const base::IntVector &config);

  virtual Buffer *allocate(size_t size);
  virtual Buffer *allocate(const BufferDesc &desc);
  virtual void deallocate(Buffer *buffer);

  virtual base::Status copy(Buffer *src, Buffer *dst);
  virtual base::Status download(Buffer *src, Buffer *dst);
  virtual base::Status upload(Buffer *src, Buffer *dst);

  virtual base::Status synchronize();

  virtual void *getContext();
  virtual void *getCommandQueue();

 protected:
  AscendCLDevice(base::DeviceType device_type, void *command_queue = nullptr,
                 std::string library_path = "")
      : Device(device_type), external_command_queue_(command_queue){};
  virtual ~AscendCLDevice(){};

  virtual base::Status init();
  virtual base::Status deinit();

 private:
  void *external_command_queue_ = nullptr;
  aclrtStream stream_ = nullptr;
  aclrtContext context_ = nullptr;
};

}  // namespace device
}  // namespace nndeploy

#endif
