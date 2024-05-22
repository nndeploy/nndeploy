

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
   * @param command_queue - command_queue (corresponding to stream under CUDA,
   * corresponding to cl::command_queue under OpenCL)
   * @param library_path - Mainly serving OpenCL, using the OpenCL dynamic
   * library provided by the user
   * @return base::Status
   */
  virtual base::Status checkDevice(int device_id = 0,
                                   void *command_queue = nullptr,
                                   std::string library_path = "") override;

  /**
   * @brief Enable the device corresponding to the current device id，mainly
   * serving GPU devices
   *
   * @param device_id - device id
   * @param command_queue - command_queue (corresponding to stream under CUDA,
   * corresponding to cl::command_queue under OpenCL)
   * @param library_path - Mainly serving OpenCL, using the OpenCL dynamic
   * library provided by the user
   * @return base::Status
   */
  virtual base::Status enableDevice(int device_id = 0,
                                    void *command_queue = nullptr,
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
  /**
   * @brief Convert MatDesc to BufferDesc.
   *
   * @param desc
   * @param config
   * @return BufferDesc
   */

  /**
   * @brief Convert TensorDesc to BufferDesc.
   *
   * @param desc
   * @param config
   * @return BufferDesc
   */
  virtual BufferDesc toBufferDesc(const TensorDesc &desc,
                                  const base::IntVector &config);
  /**
   * @brief Allocate Buffer
   *
   * @param size
   * @return Buffer*
   */
  virtual Buffer *allocate(size_t size);
  /**
   * @brief Allocate Buffer
   *
   * @param desc
   * @return Buffer*
   */
  virtual Buffer *allocate(const BufferDesc &desc);
  /**
   * @brief Deallocate buffer
   *
   * @param buffer
   */
  virtual void deallocate(Buffer *buffer);

  /**
   * @brief Copy buffer
   *
   * @param src - Device's buffer.
   * @param dst - Device's buffer.
   * @return base::Status
   * @note Ensure that the memory space of dst is greater than or equal to src.
   */
  virtual base::Status copy(Buffer *src, Buffer *dst);
  /**
   * @brief Download memory from the device to the host.
   *
   * @param src - Device's buffer.
   * @param dst - Host's buffer.
   * @return base::Status
   * @note Ensure that the memory space of dst is greater than or equal to src.
   */
  virtual base::Status download(Buffer *src, Buffer *dst);
  /**
   * @brief Upload memory from the host to the device.
   *
   * @param src - Host's buffer.
   * @param dst - Device's buffer.
   * @return base::Status
   * @note Ensure that the memory space of dst is greater than or equal to src.
   */
  virtual base::Status upload(Buffer *src, Buffer *dst);

  /**
   * @brief synchronize
   *
   * @return base::Status
   */
  virtual base::Status synchronize();

  /**
   * @brief Get the Command Queue object
   *
   * @return void*
   */
  virtual void *getCommandQueue();

 protected:
  /**
   * @brief Construct a new Cuda Device object
   *
   * @param device_type
   * @param command_queue
   * @param library_path
   */
  CudaDevice(base::DeviceType device_type, void *command_queue = nullptr,
             std::string library_path = "")
      : Device(device_type), external_command_queue_(command_queue){};
  /**
   * @brief Destroy the Cuda Device object
   *
   */
  virtual ~CudaDevice(){};

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

 private:
  void *external_command_queue_ = nullptr;
  cudaStream_t stream_;
};

}  // namespace device
}  // namespace nndeploy

#endif
