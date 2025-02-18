#ifndef _NNDEPLOY_DEVICE_ASCEND_CL_DEVICE_H_
#define _NNDEPLOY_DEVICE_ASCEND_CL_DEVICE_H_

#include "nndeploy/device/ascend_cl/ascend_cl_include.h"
#include "nndeploy/device/device.h"

namespace nndeploy {
namespace device {

class AscendCLStream;
class AscendCLEvent;

class AscendCLArchitecture : public Architecture {
 public:
  explicit AscendCLArchitecture(base::DeviceTypeCode device_type_code);

  virtual ~AscendCLArchitecture();

  void setAclConfigPath(int device_id, const std::string &acl_config_path);

  virtual base::Status checkDevice(int device_id = 0,
                                   std::string library_path = "") override;

  virtual base::Status enableDevice(int device_id = 0,
                                    std::string library_path = "") override;

  virtual Device *getDevice(int device_id) override;

  virtual std::vector<DeviceInfo> getDeviceInfo(
      std::string library_path = "") override;

 private:
  // json文件，如果要使用msprof工具分析模型各算子执行时间时需要指定，格式看ascend_cl文档
  std::map<int, std::string> acl_config_path_map_;
};

/**
 * @brief
 *
 */
class NNDEPLOY_CC_API AscendCLDevice : public Device {
  friend class AscendCLArchitecture;

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

 public:
  AscendCLDevice(base::DeviceType device_type, std::string library_path = "")
      : Device(device_type) {};
  virtual ~AscendCLDevice() {};

  void setAclConfigPath(const std::string &acl_config_path);

  virtual base::Status init();
  virtual base::Status deinit();

 private:
  aclrtContext context_ = nullptr;
  // json文件，如果要使用msprof工具分析模型各算子执行时间时需要指定，格式看ascend_cl文档
  std::string acl_config_path_ = "";
};

class AscendCLStream : public Stream {
 public:
  AscendCLStream(Device *device);
  AscendCLStream(Device *device, void *stream);
  virtual ~AscendCLStream();

  virtual base::Status synchronize();
  virtual base::Status recordEvent(Event *event);
  virtual base::Status waitEvent(Event *event);

  virtual void *getCommandQueue();

  aclrtStream getStream();

 private:
  aclrtStream stream_;
};

class AscendCLEvent : public Event {
 public:
  AscendCLEvent(Device *device);
  virtual ~AscendCLEvent();

  virtual bool queryDone();
  virtual base::Status synchronize();

  aclrtEvent getEvent();

 protected:
  aclrtEvent event_;
};

}  // namespace device
}  // namespace nndeploy

#endif