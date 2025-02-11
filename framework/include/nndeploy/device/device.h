#ifndef _NNDEPLOY_DEVICE_DEVICE_H_
#define _NNDEPLOY_DEVICE_DEVICE_H_

#include "nndeploy/base/common.h"
#include "nndeploy/base/glic_stl_include.h"
#include "nndeploy/base/log.h"
#include "nndeploy/base/macro.h"
#include "nndeploy/base/object.h"
#include "nndeploy/base/status.h"
#include "nndeploy/device/type.h"
#include "nndeploy/device/util.h"

namespace nndeploy {
namespace device {

class Device;
class Stream;
class Event;

class Buffer;

struct NNDEPLOY_CC_API DeviceInfo {
  base::DeviceType device_type_;
  bool is_support_fp16_ = false;
};

/**
 * @brief The Architecture class
 * @note 一般通过getArchitecture获取
 *
 */
class NNDEPLOY_CC_API Architecture : public base::NonCopyable {
 public:
  explicit Architecture(base::DeviceTypeCode device_type_code);

  virtual ~Architecture();

  virtual base::Status checkDevice(int device_id = 0,
                                   std::string library_path = "") = 0;

  virtual base::Status enableDevice(int device_id = 0,
                                    std::string library_path = "") = 0;

  virtual base::Status disableDevice();

  virtual Device *getDevice(int device_id) = 0;

  virtual std::vector<DeviceInfo> getDeviceInfo(
      std::string library_path = "") = 0;

  base::DeviceTypeCode getDeviceTypeCode() const;

 protected:
  std::mutex mutex_;
  /**
   * @brief device_id -> device
   *
   */
  std::map<int, Device *> devices_;

 private:
  base::DeviceTypeCode device_type_code_;
};

std::map<base::DeviceTypeCode, std::shared_ptr<Architecture>> &
getArchitectureMap();

template <typename T>
class TypeArchitectureRegister {
 public:
  explicit TypeArchitectureRegister(base::DeviceTypeCode type) {
    auto &architecture_map = getArchitectureMap();
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
  virtual base::DataFormat getDataFormatByShape(const base::IntVector &shape);

  virtual BufferDesc toBufferDesc(const TensorDesc &desc,
                                  const base::IntVector &config) = 0;

  virtual void *allocate(size_t size) = 0;
  virtual void *allocate(const BufferDesc &desc) = 0;

  virtual void deallocate(void *ptr) = 0;

  virtual void *allocatePinned(size_t size);
  virtual void *allocatePinned(const BufferDesc &desc);

  virtual void deallocatePinned(void *ptr);

  virtual base::Status copy(void *src, void *dst, size_t size,
                            Stream *stream = nullptr) = 0;
  virtual base::Status download(void *src, void *dst, size_t size,
                                Stream *stream = nullptr) = 0;
  virtual base::Status upload(void *src, void *dst, size_t size,
                              Stream *stream = nullptr) = 0;

  virtual base::Status copy(Buffer *src, Buffer *dst,
                            Stream *stream = nullptr) = 0;
  virtual base::Status download(Buffer *src, Buffer *dst,
                                Stream *stream = nullptr) = 0;
  virtual base::Status upload(Buffer *src, Buffer *dst,
                              Stream *stream = nullptr) = 0;

  // TODO: map/unmap
  // virtual Buffer* map(Buffer* src);
  // virtual base::Status unmap(Buffer* src, Buffer* dst);
  // TODO: share? opencl / vpu / hvx?
  // virtual Buffer* share(Buffer* src);
  // virtual base::Status unshare(Buffer* src, Buffer* dst);

  // context
  virtual void *getContext();

  // stream
  virtual Stream *createStream();
  virtual Stream *createStream(void *stream);
  virtual base::Status destroyStream(Stream *stream);

  // event
  virtual Event *createEvent();
  virtual base::Status destroyEvent(Event *event);
  virtual base::Status createEvents(Event **events, size_t count);
  virtual base::Status destroyEvents(Event **events, size_t count);

  base::DeviceType getDeviceType() const;

 public:
  Device(base::DeviceType device_type, std::string library_path = "")
      : device_type_(device_type) {};
  virtual ~Device() {};

  virtual base::Status init() = 0;
  virtual base::Status deinit() = 0;

 protected:
  base::DeviceType device_type_;
};

class NNDEPLOY_CC_API Stream : public base::NonCopyable {
 public:
  Stream(Device *device);
  Stream(Device *device, void *stream);

  virtual ~Stream();

  virtual base::DeviceType getDeviceType() const;
  virtual Device *getDevice() const;

  virtual base::Status synchronize();
  virtual base::Status recordEvent(Event *event);
  virtual base::Status waitEvent(Event *event);

  virtual base::Status onExecutionContextSetup();
  virtual base::Status onExecutionContextTeardown();

  virtual void *getCommandQueue();

  template <typename T>
  T *as() {
    return static_cast<T *>(this);
  }

 protected:
  bool is_external_ = false;
  Device *device_;
};

class NNDEPLOY_CC_API Event : public base::NonCopyable {
 public:
  Event(Device *device);
  virtual ~Event();

  virtual base::DeviceType getDeviceType() const;
  virtual Device *getDevice() const;

  virtual bool queryDone();
  virtual base::Status synchronize();

  template <typename T>
  T *as() {
    return static_cast<T *>(this);
  }

 protected:
  Device *device_;
};

extern NNDEPLOY_CC_API Architecture *getArchitecture(base::DeviceTypeCode type);

extern NNDEPLOY_CC_API base::DeviceType getDefaultHostDeviceType();

extern NNDEPLOY_CC_API Device *getDefaultHostDevice();

extern NNDEPLOY_CC_API bool isHostDeviceType(base::DeviceType device_type);

extern NNDEPLOY_CC_API base::Status checkDevice(base::DeviceType device_type,
                                                std::string library_path);

extern NNDEPLOY_CC_API base::Status enableDevice(base::DeviceType device_type,
                                                 std::string library_path);

extern NNDEPLOY_CC_API Device *getDevice(base::DeviceType device_type);

extern NNDEPLOY_CC_API Stream *createStream(base::DeviceType device_type);
extern NNDEPLOY_CC_API Stream *createStream(base::DeviceType device_type,
                                            void *stream);
extern NNDEPLOY_CC_API base::Status destroyStream(Stream *stream);

extern NNDEPLOY_CC_API Event *createEvent(base::DeviceType device_type);
extern NNDEPLOY_CC_API base::Status destroyEvent(Event *event);

extern NNDEPLOY_CC_API base::Status createEvents(base::DeviceType device_type,
                                                 Event **events, size_t count);
extern NNDEPLOY_CC_API base::Status destroyEvents(base::DeviceType device_type,
                                                  Event **events, size_t count);

extern NNDEPLOY_CC_API std::vector<DeviceInfo> getDeviceInfo(
    base::DeviceTypeCode type, std::string library_path);

extern NNDEPLOY_CC_API base::Status disableDevice();

extern NNDEPLOY_CC_API base::Status destoryArchitecture();

}  // namespace device
}  // namespace nndeploy

#endif