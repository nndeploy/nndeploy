/**
 * @file runtime.h
 * @author your name (you@domain.com)
 * @brief
 * @version 0.1
 * @date 2022-11-24
 *
 * @copyright Copyright (c) 2022
 *
 */
#ifndef _NNDEPLOY_INCLUDE_ARCHITECTURE_MAT_H_
#define _NNDEPLOY_INCLUDE_ARCHITECTURE_MAT_H_

#include "nndeploy/include/base/status.h"
#include "nndeploy/include/base/type.h"
#include "nndeploy/include/architecture/buffer.h"

namespace nndeploy {
namespace architecture {

class Device;

struct MatDesc {
  base::DataType data_type_;

  base::IntVector shape_;

  base::SizeVector stride_;
};

class Mat {
 public:
  Mat();
  virtual ~Mat();

  // get
  bool empty();
  base::DeviceType getDeviceType();
  int32_t getId();
  void *getPtr();

 private:
  MatDesc desc_;

  Buffer *buffer_;

  // 引用计数 + 满足多线程
};

}  // namespace architecture
}  // namespace nndeploy

#endif