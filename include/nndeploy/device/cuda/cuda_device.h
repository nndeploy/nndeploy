

#ifndef _NNDEPLOY_DEVICE_CUDA_DEVICE_H_
#define _NNDEPLOY_DEVICE_CUDA_DEVICE_H_

#include "nndeploy/device/cuda/cuda_include.h"
#include "nndeploy/device/device.h"

namespace nndeploy {
namespace device {

class CudaArchitecture : public Architecture {
 public:
  explicit CudaArchitecture(base::DeviceTypeCode device_type_code);

  virtual ~CudaArchitecture();

  virtual base::Status checkDevice(int device_id = 0,
                                   void* command_queue = NULL,
                                   std::string library_path = "") override;

  virtual base::Status enableDevice(int device_id = 0,
                                    void* command_queue = NULL,
                                    std::string library_path = "") override;

  virtual Device* getDevice(int device_id) override;

  virtual std::vector<DeviceInfo> getDeviceInfo(
      std::string library_path = "") override;
};

/**
 * @brief
 *
 */
class NNDEPLOY_CC_API CudaDevice : public Device {
  friend class CudaArchitecture;

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

  virtual base::Status synchronize();

  virtual void* getCommandQueue();

 protected:
  CudaDevice(base::DeviceType device_type, void* command_queue = NULL,
             std::string library_path = "")
      : Device(device_type), external_command_queue_(command_queue){};
  virtual ~CudaDevice(){};

  virtual base::Status init();
  virtual base::Status deinit();

 private:
  void* external_command_queue_ = NULL;
  cudaStream_t stream_;
};

}  // namespace device
}  // namespace nndeploy

#endif
