#ifndef _NNDEPLOY_DEVICE_OPENCL_OPENCL_DEVICE_H_
#define _NNDEPLOY_DEVICE_OPENCL_OPENCL_DEVICE_H_

#include "nndeploy/device/opencl/opencl_include.h"
#include "nndeploy/device/opencl/opencl_wrapper.h"
#include "nndeploy/device/device.h"

namespace nndeploy
{
namespace device
{
class OpenCLArchitecture : public Architecture
{
    public:
        /**
         * @brief
         *
         * @param device_type_code
         */
        explicit OpenCLArchitecture(base::DeviceTypeCode device_type_code);

        /**
         * @brief Destroy the OpenCL Architecture object
         *
         */
        virtual ~OpenCLArchitecture();

        /**
         * @brief Check whether the device corresponding to the current device id
         * exists
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

class NNDEPLOY_CC_API OpenCLDevice : public Device
{
    friend class OpenCLArchitecture;
    public:
        /**
         * @brief Construct a new OpenCL Device object
         *
         * @param device_type
         * @param library_path
         */
        OpenCLDevice(base::DeviceType device_type, std::string library_path = "")
            : Device(device_type){};

        /**
         * @brief Destroy the Cuda OpenCL object
         *
         */
        virtual ~OpenCLDevice(){};
                
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

}   /* namespace device */
}   /* namespace nndeploy */

#endif