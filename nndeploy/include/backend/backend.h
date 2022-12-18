/**
 * @file backend.h
 * @author your name (you@domain.com)
 * @brief
 * @version 0.1
 * @date 2022-11-24
 *
 * @copyright Copyright (c) 2022
 *
 */
#ifndef _NNDEPLOY_INCLUDE_BACKEND_BACKEND_
#define _NNDEPLOY_INCLUDE_BACKEND_BACKEND_

#include "nndeploy/include/backend/device.h"
#include "nndeploy/include/base/config.h"
#include "nndeploy/include/base/status.h"

using namespace nndeploy::base;

namespace nndeploy {
namespace backend {

struct DeviceInfo {
  bool is_support_fp16 = false;
};

class Backend {
 public:
  explicit Backend(DeviceTypeCode device_type_code);

  virtual ~Backend();

  virtual Status CheckDevice(int32_t device_id = 0,
                             std::string library_path = "") = 0;

  virtual Device *CreateDevice(int32_t device_id = 0,
                               std::string library_path = "") = 0;

  virtual Status DestoryDevice(Device *device) = 0;

  virtual std::vector<DeviceInfo> GetDeviceInfo(
      std::string library_path = "") = 0;

 private:
  DeviceTypeCode device_type_code_;
};

}  // namespace backend
}  // namespace nndeploy

#endif