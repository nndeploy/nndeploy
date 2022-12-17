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
#ifndef _NN_DEPLOY_BACKEND_BACKEND_
#define _NN_DEPLOY_BACKEND_BACKEND_

#include "nn_deploy/backend/device.h"
#include "nn_deploy/base/config.h"
#include "nn_deploy/base/status.h"

using namespace nn_deploy::base;

namespace nn_deploy {
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
}  // namespace nn_deploy

#endif