#ifndef _NNDEPLOY_DEVICE_CUDA_DEVICE_H_
#define _NNDEPLOY_DEVICE_CUDA_DEVICE_H_

#include "nndeploy/device/cuda/cuda_include.h"
#include "nndeploy/device/device.h"

namespace nndeploy {
namespace device {

class CudaArchitecture : public Architecture {
 public:
  /**
   * @brief Construct a new Cuda Architecture object
   *
   * @param device_type_code
   */
  explicit CudaArchitecture(base::DeviceTypeCode device_type_code);

  /**
   * @brief Destroy the Cuda Architecture object
   *
   */
  virtual ~CudaArchitecture();

  /**
   * @brief Check whether the device corresponding to the current device id
   * exists, mainly serving GPU devices
   *
   * @param device_id - device id
   * @param library_path - Mainly serving OpenCL, using the OpenCL dynamic
   * library provided by the user
   * @return base::Status
   */
  virtual base::Status checkDevice(int device_id = 0,
                                   std::string library_path = "") override;

  /**
   * @brief Enable the device corresponding to the current device id，mainly
   * serving GPU devices
   *
   * @param device_id - device id
   * @param library_path - Mainly serving OpenCL, using the OpenCL dynamic
   * library provided by the user
   * @return base::Status
   */
  virtual base::Status enableDevice(int device_id = 0,
                                    std::string library_path = "") override;

  /**
   * @brief Get the Device object
   *
   * @param device_id
   * @return Device*
   */
  virtual Device *getDevice(int device_id) override;

  /**
   * @brief Get the Device Info object
   *
   * @param library_path
   * @return std::vector<DeviceInfo>
   */
  virtual std::vector<DeviceInfo> getDeviceInfo(
      std::string library_path = "") override;
};

/**
 * @brief
 *
 */
class NNDEPLOY_CC_API CudaDevice : public Device {
  /**
   * @brief friend class
   *
   */
  friend class CudaArchitecture;

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

  // TODO: map/unmap
  // virtual Buffer* map(Buffer* src);
  // virtual base::Status unmap(Buffer* src, Buffer* dst);
  // TODO: share? opencl / vpu / hvx?
  // virtual Buffer* share(Buffer* src);
  // virtual base::Status unshare(Buffer* src, Buffer* dst);

  virtual void *getContext();

  // stream
  virtual Stream *createStream();
  virtual Stream *createStream(void *stream);
  virtual base::Status deleteStream(Stream *stream);

  // event
  virtual Event *createEvent();
  virtual base::Status destroyEvent(Event *event);
  virtual base::Status createEvents(Event **events, size_t count);
  virtual base::Status destroyEvents(Event **events, size_t count);

 public:
  /**
   * @brief Construct a new Cuda Device object
   *
   * @param device_type
   * @param stream
   * @param library_path
   */
  CudaDevice(base::DeviceType device_type, std::string library_path = "")
      : Device(device_type) {};
  /**
   * @brief Destroy the Cuda Device object
   *
   */
  virtual ~CudaDevice() {};

  /**
   * @brief init
   *
   * @return base::Status
   */
  virtual base::Status init();
  /**
   * @brief deinit
   *
   * @return base::Status
   */
  virtual base::Status deinit();
};

class CudaStream : public Stream {
 public:
  CudaStream(Device *device);
  CudaStream(Device *device, void *stream);
  virtual ~CudaStream();

  virtual base::Status synchronize();
  virtual base::Status recordEvent(Event *event);
  virtual base::Status waitEvent(Event *event);

  cudaStream_t getStream();

 private:
  cudaStream_t stream_;
};

class CudaEvent : public Event {
 public:
  CudaEvent(Device *device);
  virtual ~CudaEvent();

  virtual bool queryDone();
  virtual base::Status synchronize();

  cudaEvent_t getEvent();

 protected:
  cudaEvent_t event_;
};

}  // namespace device
}  // namespace nndeploy

#endif
