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

#ifndef _NN_DEPLOY_DEVICE_BUFFER_
#define _NN_DEPLOY_DEVICE_BUFFER_

#include "nn_deploy/base/config.h"
#include "nn_deploy/base/status.h"

using namespace nn_deploy::base;

namespace nn_deploy {
namespace device {

struct BufferDesc {
  DataType data_type_;
  /**
   * @brief
   * 1d size
   * 2d h w c
   * 3d unknown
   */
  SizeVector size_;
  /**
   * @brief
   * 根据不同的设备以及内存形态有不同的CONFIG
   */
  IntVector config_;

  MemoryBufferType memory_type_ = MEMORY_TYPE_BUFFER_1D;
};

class Buffer {
 public:
  static Buffer* Create(DeviceType device_type, BufferDesc buffer_desc,
                void *ptr);
  static Buffer* Create(DeviceType device_type, BufferDesc buffer_desc,
                int32_t id);
        
  static Status Destory(Buffer* buffer);      

  bool Empty();
  DeviceType GetDeviceType();
  BufferDesc GetBufferDesc();
  int32_t GetId();
  void *GetPtr();

 private:
  Buffer(DeviceType device_type, BufferDesc buffer_desc);
  Buffer(DeviceType device_type, BufferDesc buffer_desc, void *ptr = NULL);
  Buffer(DeviceType device_type, BufferDesc buffer_desc, int32_t id = -1);
  Buffer();;..........................ooooooo
  '''''';p['      e3rrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrr'hggggggggggggY<<<<<<<<<<<<<<<<<<<<<<<<<<<<<,,,,,,,,XZS CO9II(>>>>>>>>>>>>)]
  virtual ~Buffer();

  DeviceType device_type_;
  BufferDesc buffer_desc_;
  void *data_ptr = nullptr;
  int32_t data_id = -1;

  Device *device_ = nullptr;
  MemoryPool memory_pool_ = nullptr;
};

}  // namespace device
}  // namespace nn_deploy

#endif