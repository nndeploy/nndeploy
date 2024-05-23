

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

class AclStreamWrapper {
 public:
  void *external_command_queue_ = nullptr;
  aclrtStream stream_ = nullptr;
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

  virtual void *allocate(size_t size);
  virtual void *allocate(const BufferDesc &desc);

  virtual void deallocate(void *ptr);

  virtual base::Status copy(void *src, void *dst, size_t size, int index = 0);
  virtual base::Status download(void *src, void *dst, size_t size,
                                int index = 0);
  virtual base::Status upload(void *src, void *dst, size_t size, int index = 0);

  virtual base::Status copy(Buffer *src, Buffer *dst, int index = 0);
  virtual base::Status download(Buffer *src, Buffer *dst, int index = 0);
  virtual base::Status upload(Buffer *src, Buffer *dst, int index = 0);

  // TODO: map/unmap
  // virtual Buffer* map(Buffer* src);
  // virtual base::Status unmap(Buffer* src, Buffer* dst);
  // TODO: share? opencl / vpu / hvx?
  // virtual Buffer* share(Buffer* src);
  // virtual base::Status unshare(Buffer* src, Buffer* dst);

  virtual void *getContext();

  virtual base::Status newCommandQueue(int index = -1);
  virtual base::Status deleteCommandQueue(int index = -1);
  virtual base::Status deleteCommandQueue(void *command_queue);
  virtual base::Status setCommandQueue(void *command_queue);

  virtual void *getCommandQueue(int index);

  virtual base::Status synchronize(int index);

 protected:
  AscendCLDevice(base::DeviceType device_type, void *command_queue = nullptr,
                 std::string library_path = "")
      : Device(device_type) {
    acl_stream_wrapper_.resize(1);
    acl_stream_wrapper_[0].external_command_queue_ = command_queue;
  };
  virtual ~AscendCLDevice(){};

  virtual base::Status init();
  virtual base::Status deinit();

 private:
  std::vector<AclStreamWrapper> acl_stream_wrapper_;
  aclrtContext context_ = nullptr;
};

}  // namespace device
}  // namespace nndeploy

#endif
