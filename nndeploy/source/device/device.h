
#ifndef _NNDEPLOY_SOURCE_DEVICE_DEVICE_H_
#define _NNDEPLOY_SOURCE_DEVICE_DEVICE_H_

#include "nndeploy/source/base/basic.h"
#include "nndeploy/source/base/glic_stl_include.h"
#include "nndeploy/source/base/log.h"
#include "nndeploy/source/base/macro.h"
#include "nndeploy/source/base/object.h"
#include "nndeploy/source/base/status.h"

namespace nndeploy {
namespace device {

struct BufferDesc;

struct MatDesc;

struct TensorDesc;

class Buffer;

class Device;

/**
 * @brief buffer的内存来源类型
 */
enum BufferSourceType : int32_t {
  kBufferSourceTypeNone = 0x0000,
  kBufferSourceTypeAllocate,
  kBufferSourceTypeExternal,
  kBufferSourceTypeMapped,
};

struct NNDEPLOY_CC_API DeviceInfo {
  base::DeviceType device_type_;
  bool is_support_fp16_ = false;
};

/**
 * @brief The Architecture class
 * @note 不可以new，只能通过getArchitecture获取
 *
 */
class NNDEPLOY_CC_API Architecture : public base::NonCopyable {
 public:
  explicit Architecture(base::DeviceTypeCode device_type_code);

  virtual ~Architecture();

  virtual base::Status checkDevice(int32_t device_id = 0,
                                   void* command_queue = NULL,
                                   std::string library_path = "") = 0;

  virtual base::Status enableDevice(int32_t device_id = 0,
                                    void* command_queue = NULL,
                                    std::string library_path = "") = 0;

  virtual Device* getDevice(int32_t device_id) = 0;

  virtual std::vector<DeviceInfo> getDeviceInfo(
      std::string library_path = "") = 0;

  base::DeviceTypeCode getDeviceTypeCode();

 protected:
  std::mutex mutex_;
  std::map<int32_t, Device*> devices_;

 private:
  base::DeviceTypeCode device_type_code_;
};

std::map<base::DeviceTypeCode, std::shared_ptr<Architecture>>&
getArchitectureMap();

template <typename T>
class TypeArchitectureRegister {
 public:
  explicit TypeArchitectureRegister(base::DeviceTypeCode type) {
    auto& architecture_map = getArchitectureMap();
    if (architecture_map.find(type) == architecture_map.end()) {
      architecture_map[type] = std::shared_ptr<T>(new T(type));
    }
  }
};

/**
 * @brief
 *
 */
class NNDEPLOY_CC_API Device : public base::NonCopyable {
  friend class Architecture;

 public:
  virtual BufferDesc& toBufferDesc(const MatDesc& desc,
                                   const base::IntVector& config) = 0;

  virtual BufferDesc& toBufferDesc(const TensorDesc& desc,
                                   const base::IntVector& config) = 0;

  virtual Buffer* allocate(size_t size) = 0;
  virtual Buffer* allocate(const BufferDesc& desc) = 0;
  virtual void deallocate(Buffer* buffer) = 0;

  Buffer* create(
      size_t size, void* ptr,
      BufferSourceType buffer_source_type = kBufferSourceTypeExternal);
  Buffer* create(
      const BufferDesc& desc, void* ptr,
      BufferSourceType buffer_source_type = kBufferSourceTypeExternal);
  Buffer* create(
      size_t size, int32_t id,
      BufferSourceType buffer_source_type = kBufferSourceTypeExternal);
  Buffer* create(
      const BufferDesc& desc, int32_t id,
      BufferSourceType buffer_source_type = kBufferSourceTypeExternal);

  virtual base::Status copy(Buffer* src, Buffer* dst) = 0;
  virtual base::Status download(Buffer* src, Buffer* dst) = 0;
  virtual base::Status upload(Buffer* src, Buffer* dst) = 0;
  // TODO: map/unmap
  // virtual Buffer* map(Buffer* src);
  // virtual base::Status unmap(Buffer* src, Buffer* dst);
  // TODO: share? opencl / vpu / hvx?
  // virtual Buffer* share(Buffer* src);
  // virtual base::Status unshare(Buffer* src, Buffer* dst);

  virtual base::Status synchronize();

  virtual void* getCommandQueue();

  base::DeviceType getDeviceType();

 protected:
  Device(base::DeviceType device_type, void* command_queue = NULL,
         std::string library_path = "")
      : device_type_(device_type){};
  virtual ~Device(){};

  virtual base::Status init() = 0;
  virtual base::Status deinit() = 0;

  void destory(Buffer* buffer);

 protected:
  base::DeviceType device_type_;
};

extern NNDEPLOY_CC_API Architecture* getArchitecture(base::DeviceTypeCode type);

extern NNDEPLOY_CC_API base::DeviceType getDefaultHostDeviceType();

extern NNDEPLOY_CC_API Device* getDefaultHostDevice();

extern NNDEPLOY_CC_API bool isHostDeviceType(base::DeviceType device_type);

extern NNDEPLOY_CC_API base::Status checkDevice(base::DeviceType device_type,
                                                void* command_queue,
                                                std::string library_path);

extern NNDEPLOY_CC_API base::Status enableDevice(base::DeviceType device_type,
                                                 void* command_queue,
                                                 std::string library_path);

extern NNDEPLOY_CC_API Device* getDevice(base::DeviceType device_type);

extern NNDEPLOY_CC_API std::vector<DeviceInfo> getDeviceInfo(
    base::DeviceTypeCode type, std::string library_path);

}  // namespace device
}  // namespace nndeploy

#endif