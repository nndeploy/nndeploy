
#ifndef _NNDEPLOY_DEVICE_ARM_DEVICE_H_
#define _NNDEPLOY_DEVICE_ARM_DEVICE_H_

#include "nndeploy/device/device.h"

namespace nndeploy {
namespace device {

class ArmArchitecture : public Architecture {
 public:
  /**
   * @brief Construct a new Arm Architecture object
   *
   * @param device_type_code
   */
  explicit ArmArchitecture(base::DeviceTypeCode device_type_code);

  /**
   * @brief Destroy the Arm Architecture object
   *
   */
  virtual ~ArmArchitecture();

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
class NNDEPLOY_CC_API ArmDevice : public Device {
  friend class ArmArchitecture;

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

 protected:
  ArmDevice(base::DeviceType device_type, void *command_queue = nullptr,
            std::string library_path = "")
      : Device(device_type){};
  virtual ~ArmDevice(){};

  virtual base::Status init();
  virtual base::Status deinit();
};

}  // namespace device
}  // namespace nndeploy

#endif
