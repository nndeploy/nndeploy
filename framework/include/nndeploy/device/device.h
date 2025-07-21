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

/**
 * @brief 设备信息结构体
 * 
 * 用于描述设备的基本属性信息
 */
struct NNDEPLOY_CC_API DeviceInfo {
  base::DeviceType device_type_;   ///< 设备类型
  bool is_support_fp16_ = false;   ///< 是否支持半精度浮点数
};

/**
 * @brief 设备架构类
 * 
 * 管理特定类型设备的抽象基类，负责设备的创建、管理和销毁。
 * 一般通过 getArchitecture() 函数获取实例。
 * 
 * @note 一般通过getArchitecture获取
 */
class NNDEPLOY_CC_API Architecture : public base::NonCopyable {
 public:
  /**
   * @brief 构造函数
   * @param device_type_code 设备类型代码
   */
  explicit Architecture(base::DeviceTypeCode device_type_code);

  /**
   * @brief 析构函数
   */
  virtual ~Architecture();

  /**
   * @brief 检查设备是否可用
   * @param device_id 设备ID，默认为0
   * @param library_path 库文件路径，默认为空
   * @return base::Status 检查结果状态
   */
  virtual base::Status checkDevice(int device_id = 0,
                                   std::string library_path = "") = 0;

  /**
   * @brief 启用设备
   * @param device_id 设备ID，默认为0
   * @param library_path 库文件路径，默认为空
   * @return base::Status 启用结果状态
   */
  virtual base::Status enableDevice(int device_id = 0,
                                    std::string library_path = "") = 0;

  /**
   * @brief 禁用设备
   * @return base::Status 禁用结果状态
   */
  virtual base::Status disableDevice();

  /**
   * @brief 获取指定ID的设备
   * @param device_id 设备ID
   * @return Device* 设备指针
   */
  virtual Device *getDevice(int device_id) = 0;

  /**
   * @brief 获取设备信息列表
   * @param library_path 库文件路径，默认为空
   * @return std::vector<DeviceInfo> 设备信息列表
   */
  virtual std::vector<DeviceInfo> getDeviceInfo(
      std::string library_path = "") = 0;

  /**
   * @brief 获取设备类型代码
   * @return base::DeviceTypeCode 设备类型代码
   */
  base::DeviceTypeCode getDeviceTypeCode() const;

  /**
   * @brief 插入设备（供Python使用）
   * @param device_id 设备ID
   * @param device 设备指针
   * @return base::Status 插入结果状态
   */
  base::Status insertDevice(int device_id, Device *device);

 protected:
  std::mutex mutex_;                       ///< 线程安全锁
  /**
   * @brief 设备映射表
   * 
   * 从设备ID到设备指针的映射
   */
  std::map<int, Device *> devices_;

 private:
  base::DeviceTypeCode device_type_code_;  ///< 设备类型代码
};

/**
 * @brief 获取架构映射表
 * @return std::map<base::DeviceTypeCode, std::shared_ptr<Architecture>>& 架构映射表的引用
 */
extern NNDEPLOY_CC_API std::map<base::DeviceTypeCode,
                                 std::shared_ptr<Architecture>> &
    getArchitectureMap();

/**
 * @brief 类型架构注册器模板类
 * 
 * 用于自动注册特定类型的设备架构
 * 
 * @tparam T 架构类型
 */
template <typename T>
class TypeArchitectureRegister {
 public:
  /**
   * @brief 构造函数，自动注册架构
   * @param type 设备类型代码
   */
  explicit TypeArchitectureRegister(base::DeviceTypeCode type) {
    auto &architecture_map = getArchitectureMap();
    if (architecture_map.find(type) == architecture_map.end()) {
      architecture_map[type] = std::shared_ptr<T>(new T(type));
    }
  }
};

/**
 * @brief 设备抽象基类
 * 
 * 定义了设备的基本接口，包括内存管理、数据传输、流和事件管理等功能
 */
class NNDEPLOY_CC_API Device : public base::NonCopyable {
  friend class Architecture;

 public:
  /**
   * @brief 根据形状获取数据格式
   * @param shape 张量形状
   * @return base::DataFormat 数据格式
   */
  virtual base::DataFormat getDataFormatByShape(const base::IntVector &shape);

  /**
   * @brief 转换为缓冲区描述符
   * @param desc 张量描述符
   * @param config 配置参数
   * @return BufferDesc 缓冲区描述符
   */
  virtual BufferDesc toBufferDesc(const TensorDesc &desc,
                                  const base::IntVector &config) = 0;

  /**
   * @brief 分配内存
   * @param size 内存大小（字节）
   * @return void* 分配的内存指针
   */
  virtual void *allocate(size_t size) = 0;
  
  /**
   * @brief 分配内存
   * @param desc 缓冲区描述符
   * @return void* 分配的内存指针
   */
  virtual void *allocate(const BufferDesc &desc) = 0;

  /**
   * @brief 释放内存
   * @param ptr 要释放的内存指针
   */
  virtual void deallocate(void *ptr) = 0;

  /**
   * @brief 分配锁页内存
   * @param size 内存大小（字节）
   * @return void* 分配的锁页内存指针
   */
  virtual void *allocatePinned(size_t size);
  
  /**
   * @brief 分配锁页内存
   * @param desc 缓冲区描述符
   * @return void* 分配的锁页内存指针
   */
  virtual void *allocatePinned(const BufferDesc &desc);

  /**
   * @brief 释放锁页内存
   * @param ptr 要释放的锁页内存指针
   */
  virtual void deallocatePinned(void *ptr);

  /**
   * @brief 内存拷贝
   * @param src 源内存指针
   * @param dst 目标内存指针
   * @param size 拷贝大小（字节）
   * @param stream 执行流，默认为nullptr
   * @return base::Status 拷贝结果状态
   */
  virtual base::Status copy(void *src, void *dst, size_t size,
                            Stream *stream = nullptr) = 0;
  
  /**
   * @brief 从设备下载数据到主机
   * @param src 设备内存指针
   * @param dst 主机内存指针
   * @param size 传输大小（字节）
   * @param stream 执行流，默认为nullptr
   * @return base::Status 下载结果状态
   */
  virtual base::Status download(void *src, void *dst, size_t size,
                                Stream *stream = nullptr) = 0;
  
  /**
   * @brief 从主机上传数据到设备
   * @param src 主机内存指针
   * @param dst 设备内存指针
   * @param size 传输大小（字节）
   * @param stream 执行流，默认为nullptr
   * @return base::Status 上传结果状态
   */
  virtual base::Status upload(void *src, void *dst, size_t size,
                              Stream *stream = nullptr) = 0;

  /**
   * @brief 缓冲区拷贝
   * @param src 源缓冲区
   * @param dst 目标缓冲区
   * @param stream 执行流，默认为nullptr
   * @return base::Status 拷贝结果状态
   */
  virtual base::Status copy(Buffer *src, Buffer *dst,
                            Stream *stream = nullptr) = 0;
  
  /**
   * @brief 缓冲区下载
   * @param src 源缓冲区
   * @param dst 目标缓冲区
   * @param stream 执行流，默认为nullptr
   * @return base::Status 下载结果状态
   */
  virtual base::Status download(Buffer *src, Buffer *dst,
                                Stream *stream = nullptr) = 0;
  
  /**
   * @brief 缓冲区上传
   * @param src 源缓冲区
   * @param dst 目标缓冲区
   * @param stream 执行流，默认为nullptr
   * @return base::Status 上传结果状态
   */
  virtual base::Status upload(Buffer *src, Buffer *dst,
                              Stream *stream = nullptr) = 0;

  // TODO: map/unmap
  // virtual Buffer* map(Buffer* src);
  // virtual base::Status unmap(Buffer* src, Buffer* dst);
  // TODO: share? opencl / vpu / hvx?
  // virtual Buffer* share(Buffer* src);
  // virtual base::Status unshare(Buffer* src, Buffer* dst);

  /**
   * @brief 获取设备上下文
   * @return void* 设备上下文指针
   */
  virtual void *getContext();
  
  /**
   * @brief 绑定线程
   * @return base::Status 绑定结果状态
   */
  virtual base::Status bindThread();

  /**
   * @brief 创建流
   * @return Stream* 流指针
   */
  virtual Stream *createStream();
  
  /**
   * @brief 从现有流创建流对象
   * @param stream 原生流指针
   * @return Stream* 流指针
   */
  virtual Stream *createStream(void *stream);
  
  /**
   * @brief 销毁流
   * @param stream 要销毁的流
   * @return base::Status 销毁结果状态
   */
  virtual base::Status destroyStream(Stream *stream);

  /**
   * @brief 创建事件
   * @return Event* 事件指针
   */
  virtual Event *createEvent();
  
  /**
   * @brief 销毁事件
   * @param event 要销毁的事件
   * @return base::Status 销毁结果状态
   */
  virtual base::Status destroyEvent(Event *event);
  
  /**
   * @brief 批量创建事件
   * @param events 事件指针数组
   * @param count 事件数量
   * @return base::Status 创建结果状态
   */
  virtual base::Status createEvents(Event **events, size_t count);
  
  /**
   * @brief 批量销毁事件
   * @param events 事件指针数组
   * @param count 事件数量
   * @return base::Status 销毁结果状态
   */
  virtual base::Status destroyEvents(Event **events, size_t count);

  /**
   * @brief 获取设备类型
   * @return base::DeviceType 设备类型
   */
  base::DeviceType getDeviceType() const;

 public:
  /**
   * @brief 构造函数
   * @param device_type 设备类型
   * @param library_path 库文件路径，默认为空
   */
  Device(base::DeviceType device_type, std::string library_path = "")
      : device_type_(device_type) {};
  
  /**
   * @brief 析构函数
   */
  virtual ~Device() {};

  /**
   * @brief 初始化设备
   * @return base::Status 初始化结果状态
   */
  virtual base::Status init() = 0;
  
  /**
   * @brief 反初始化设备
   * @return base::Status 反初始化结果状态
   */
  virtual base::Status deinit() = 0;

 protected:
  base::DeviceType device_type_;  ///< 设备类型
};

/**
 * @brief 流类
 * 
 * 管理设备上的执行流，用于异步操作和同步控制
 */
class NNDEPLOY_CC_API Stream : public base::NonCopyable {
 public:
  /**
   * @brief 构造函数
   * @param device 关联的设备
   */
  Stream(Device *device);
  
  /**
   * @brief 构造函数（从现有流）
   * @param device 关联的设备
   * @param stream 现有的原生流
   */
  Stream(Device *device, void *stream);

  /**
   * @brief 析构函数
   */
  virtual ~Stream();

  /**
   * @brief 获取设备类型
   * @return base::DeviceType 设备类型
   */
  virtual base::DeviceType getDeviceType() const;
  
  /**
   * @brief 获取关联设备
   * @return Device* 设备指针
   */
  virtual Device *getDevice() const;

  /**
   * @brief 同步流
   * @return base::Status 同步结果状态
   */
  virtual base::Status synchronize();
  
  /**
   * @brief 在流中记录事件
   * 
   * 在当前流的执行位置记录一个事件，可用于后续的同步操作。
   * 
   * 示例:
   * ```cpp
   * Stream *stream = device->createStream();
   * Event *event = device->createEvent();
   *
   * // 执行一些操作
   * kernel1<<<...>>>(stream->getNativeStream());
   *
   * // 在当前位置记录事件
   * stream->recordEvent(event);
   *
   * // 继续执行其他操作
   * kernel2<<<...>>>(stream->getNativeStream());
   * ```
   * 
   * @param event 要记录的事件对象指针
   * @return base::Status 操作状态，成功返回kStatusCodeOk
   */
  virtual base::Status recordEvent(Event *event);

  /**
   * @brief 等待事件完成
   * 
   * 让当前流等待指定事件完成后再继续执行后续操作。
   * 
   * 示例:
   * ```cpp
   * Stream *stream1 = device->createStream();
   * Stream *stream2 = device->createStream();
   * Event *event = device->createEvent();
   *
   * // 在stream1中执行操作并记录事件
   * kernel1<<<...>>>(stream1->getNativeStream());
   * stream1->recordEvent(event);
   *
   * // stream2等待stream1中的event完成后才执行
   * stream2->waitEvent(event);
   * kernel2<<<...>>>(stream2->getNativeStream());
   * ```
   * 
   * @param event 要等待的事件对象指针
   * @return base::Status 操作状态，成功返回kStatusCodeOk
   */
  virtual base::Status waitEvent(Event *event);

  /**
   * @brief 执行上下文设置
   * @return base::Status 设置结果状态
   */
  virtual base::Status onExecutionContextSetup();
  
  /**
   * @brief 执行上下文清理
   * @return base::Status 清理结果状态
   */
  virtual base::Status onExecutionContextTeardown();

  /**
   * @brief 获取原生流对象
   * @return void* 原生流指针
   */
  virtual void *getNativeStream();

  /**
   * @brief 类型转换模板函数
   * @tparam T 目标类型
   * @return T* 转换后的指针
   */
  template <typename T>
  T *as() {
    return static_cast<T *>(this);
  }

 protected:
  bool is_external_ = false;  ///< 是否为外部流
  Device *device_;            ///< 关联的设备
};

/**
 * @brief 事件类
 * 
 * 用于流之间的同步和异步操作的完成状态查询
 */
class NNDEPLOY_CC_API Event : public base::NonCopyable {
 public:
  /**
   * @brief 构造函数
   * @param device 关联的设备
   */
  Event(Device *device);
  
  /**
   * @brief 析构函数
   */
  virtual ~Event();

  /**
   * @brief 获取设备类型
   * @return base::DeviceType 设备类型
   */
  virtual base::DeviceType getDeviceType() const;
  
  /**
   * @brief 获取关联设备
   * @return Device* 设备指针
   */
  virtual Device *getDevice() const;

  /**
   * @brief 查询事件是否已完成
   *
   * 非阻塞地检查事件是否已经完成执行。这个方法允许应用程序在不阻塞当前线程的情况下
   * 检查事件状态。
   * 
   * 示例:
   * ```cpp
   * Event *event = device->createEvent();
   * stream->recordEvent(event);
   *
   * // 稍后检查事件是否完成
   * if (event->queryDone()) {
   *   // 事件已完成，可以安全地访问相关资源
   * }
   * ```
   * 
   * @return bool 如果事件已完成返回true，否则返回false
   */
  virtual bool queryDone();

  /**
   * @brief 同步等待事件完成
   * 
   * 阻塞当前线程直到事件完成。这个方法用于确保在继续执行之前，
   * 与事件相关的所有操作都已完成。
   * 
   * 示例:
   * ```cpp
   * Event *event = device->createEvent();
   * stream->recordEvent(event);
   *
   * // 等待事件完成
   * event->synchronize();
   *
   * // 此时可以安全地访问相关资源
   * ```
   * 
   * @return base::Status 操作状态，成功返回kStatusCodeOk
   */
  virtual base::Status synchronize();

  /**
   * @brief 获取原生事件对象
   * @return void* 原生事件指针
   */
  virtual void *getNativeEvent();

  /**
   * @brief 类型转换模板函数
   * @tparam T 目标类型
   * @return T* 转换后的指针
   */
  template <typename T>
  T *as() {
    return static_cast<T *>(this);
  }

 protected:
  Device *device_;  ///< 关联的设备
};

/**
 * @brief 获取指定类型的架构
 * @param type 设备类型代码
 * @return Architecture* 架构指针
 */
extern NNDEPLOY_CC_API Architecture *getArchitecture(base::DeviceTypeCode type);

/**
 * @brief 获取指定类型架构的共享指针
 * @param type 设备类型代码
 * @return std::shared_ptr<Architecture> 架构共享指针
 */
extern NNDEPLOY_CC_API std::shared_ptr<Architecture> getArchitectureSharedPtr(
    base::DeviceTypeCode type);

/**
 * @brief 获取默认主机设备类型
 * @return base::DeviceType 默认主机设备类型
 */
extern NNDEPLOY_CC_API base::DeviceType getDefaultHostDeviceType();

/**
 * @brief 获取默认主机设备
 * @return Device* 默认主机设备指针
 */
extern NNDEPLOY_CC_API Device *getDefaultHostDevice();

/**
 * @brief 判断是否为主机设备类型
 * @param device_type 设备类型
 * @return bool 是主机设备类型返回true，否则返回false
 */
extern NNDEPLOY_CC_API bool isHostDeviceType(base::DeviceType device_type);

/**
 * @brief 检查设备是否可用
 * @param device_type 设备类型
 * @param library_path 库文件路径
 * @return base::Status 检查结果状态
 */
extern NNDEPLOY_CC_API base::Status checkDevice(base::DeviceType device_type,
                                                std::string library_path);

/**
 * @brief 启用设备
 * @param device_type 设备类型
 * @param library_path 库文件路径
 * @return base::Status 启用结果状态
 */
extern NNDEPLOY_CC_API base::Status enableDevice(base::DeviceType device_type,
                                                 std::string library_path);

/**
 * @brief 获取指定类型的设备
 * @param device_type 设备类型
 * @return Device* 设备指针
 */
extern NNDEPLOY_CC_API Device *getDevice(base::DeviceType device_type);

/**
 * @brief 创建指定类型的流
 * @param device_type 设备类型
 * @return Stream* 流指针
 */
extern NNDEPLOY_CC_API Stream *createStream(base::DeviceType device_type);

/**
 * @brief 从现有流创建流对象
 * @param device_type 设备类型
 * @param stream 原生流指针
 * @return Stream* 流指针
 */
extern NNDEPLOY_CC_API Stream *createStream(base::DeviceType device_type,
                                            void *stream);

/**
 * @brief 销毁流
 * @param stream 要销毁的流
 * @return base::Status 销毁结果状态
 */
extern NNDEPLOY_CC_API base::Status destroyStream(Stream *stream);

/**
 * @brief 创建指定类型的事件
 * @param device_type 设备类型
 * @return Event* 事件指针
 */
extern NNDEPLOY_CC_API Event *createEvent(base::DeviceType device_type);

/**
 * @brief 销毁事件
 * @param event 要销毁的事件
 * @return base::Status 销毁结果状态
 */
extern NNDEPLOY_CC_API base::Status destroyEvent(Event *event);

/**
 * @brief 批量创建事件
 * @param device_type 设备类型
 * @param events 事件指针数组
 * @param count 事件数量
 * @return base::Status 创建结果状态
 */
extern NNDEPLOY_CC_API base::Status createEvents(base::DeviceType device_type,
                                                 Event **events, size_t count);

/**
 * @brief 批量销毁事件
 * @param device_type 设备类型
 * @param events 事件指针数组
 * @param count 事件数量
 * @return base::Status 销毁结果状态
 */
extern NNDEPLOY_CC_API base::Status destroyEvents(base::DeviceType device_type,
                                                  Event **events, size_t count);

/**
 * @brief 获取设备信息
 * @param type 设备类型代码
 * @param library_path 库文件路径
 * @return std::vector<DeviceInfo> 设备信息列表
 */
extern NNDEPLOY_CC_API std::vector<DeviceInfo> getDeviceInfo(
    base::DeviceTypeCode type, std::string library_path);

/**
 * @brief 禁用设备
 * @return base::Status 禁用结果状态
 */
extern NNDEPLOY_CC_API base::Status disableDevice();

/**
 * @brief 销毁架构
 * @return base::Status 销毁结果状态
 */
extern NNDEPLOY_CC_API base::Status destoryArchitecture();

}  // namespace device
}  // namespace nndeploy

#endif