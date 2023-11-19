
#ifndef _NNDEPLOY_DEVICE_TENSOR_H_
#define _NNDEPLOY_DEVICE_TENSOR_H_

#include "nndeploy/base/common.h"
#include "nndeploy/base/glic_stl_include.h"
#include "nndeploy/base/log.h"
#include "nndeploy/base/macro.h"
#include "nndeploy/base/object.h"
#include "nndeploy/base/status.h"
#include "nndeploy/device/buffer.h"
#include "nndeploy/device/buffer_pool.h"
#include "nndeploy/device/device.h"

namespace nndeploy {
namespace device {

/**
 * @brief 描述tensor的信息
 *
 */
struct NNDEPLOY_CC_API TensorDesc {
  TensorDesc(){};
  explicit TensorDesc(base::DataType data_type, base::DataFormat format,
                      const base::IntVector &shape,
                      const base::SizeVector &stride)
      : data_type_(data_type),
        data_format_(format),
        shape_(shape),
        stride_(stride){};

  TensorDesc(const TensorDesc &desc) {
    this->data_type_ = desc.data_type_;
    this->data_format_ = desc.data_format_;
    this->shape_ = desc.shape_;
    this->stride_ = desc.stride_;
  };
  TensorDesc &operator=(const TensorDesc &desc) = default;

  virtual ~TensorDesc(){};

  bool operator==(const TensorDesc &other) {
    bool flag0 = std::equal(shape_.begin(), shape_.end(), other.shape_.begin(),
                            other.shape_.end());
    bool flag1 = std::equal(stride_.begin(), stride_.end(),
                            other.stride_.begin(), other.stride_.end());
    bool flag2 = data_type_ == other.data_type_;
    bool flag3 = data_format_ == other.data_format_;
    return flag0 && flag1 && flag2 && flag3;
  }
  bool operator!=(const TensorDesc &other) { return !(*this == other); }

  base::DataType data_type_ = base::dataTypeOf<float>();        // 数据类型
  base::DataFormat data_format_ = base::kDataFormatNotSupport;  // 数据格式
  base::IntVector shape_;                                       // 数据形状
  base::SizeVector stride_;                                     // 数据步长
};

/**
 * @brief Tensor类
 *
 */
class NNDEPLOY_CC_API Tensor : public base::NonCopyable {
 public:
  /**
   * @brief Construct a new Tensor object
   *
   */
  Tensor();
  /**
   * @brief Deconstruct the Tensor object
   *
   */
  virtual ~Tensor();

  /**
   * @brief Construct a new Tensor object
   *
   * @param name
   */
  Tensor(const std::string &name);

  /**
   * @brief Construct a new Tensor object
   *
   * @param desc
   * @param name
   */
  Tensor(const TensorDesc &desc, const std::string &name = "");
  /**
   * @brief Construct a new Tensor object with device
   *
   * @param device
   * @param desc
   * @param name
   * @param config
   */
  Tensor(Device *device, const TensorDesc &desc, const std::string &name = "",
         const base::IntVector &config = base::IntVector());
  /**
   * @brief Construct a new Tensor object with device and data_ptr
   *
   * @param device
   * @param desc
   * @param data_ptr
   * @param name
   * @param config
   */
  Tensor(Device *device, const TensorDesc &desc, void *data_ptr,
         const std::string &name = "",
         const base::IntVector &config = base::IntVector());
  /**
   * @brief Construct a new Tensor object with device and data_id
   *
   * @param device
   * @param desc
   * @param data_id
   * @param name
   * @param config
   */
  Tensor(Device *device, const TensorDesc &desc, int data_id,
         const std::string &name = "",
         const base::IntVector &config = base::IntVector());
  /**
   * @brief Construct a new Tensor object with buffer
   *
   * @param desc
   * @param buffer
   * @param name
   */
  Tensor(const TensorDesc &desc, Buffer *buffer, const std::string &name = "");

  /**
   * @brief 创建一个tensor
   *
   * @param desc
   * @param name
   * @details Tensor必须为空
   */
  void create(const TensorDesc &desc, const std::string &name);
  /**
   * @brief 创建一个tensor
   *
   * @param desc
   * @details Tensor必须为空
   */
  void create(const TensorDesc &desc);
  /**
   * @brief 创建一个tensor
   *
   * @param device
   * @param desc
   * @param name
   * @param config
   * @details Tensor必须为空
   */
  void create(Device *device, const TensorDesc &desc, const std::string &name,
              const base::IntVector &config = base::IntVector());
  /**
   * @brief 创建一个tensor
   *
   * @param device
   * @param desc
   * @param config
   * @details Tensor必须为空
   */
  void create(Device *device, const TensorDesc &desc,
              const base::IntVector &config = base::IntVector());
  /**
   * @brief 创建一个tensor
   *
   * @param device
   * @param desc
   * @param data_ptr
   * @param name
   * @param config
   * @details Tensor必须为空
   */
  void create(Device *device, const TensorDesc &desc, void *data_ptr,
              const std::string &name,
              const base::IntVector &config = base::IntVector());
  /**
   * @brief 创建一个tensor
   *
   * @param device
   * @param desc
   * @param data_ptr
   * @param config
   * @details Tensor必须为空
   */
  void create(Device *device, const TensorDesc &desc, void *data_ptr,
              const base::IntVector &config = base::IntVector());
  /**
   * @brief 创建一个tensor
   *
   * @param device
   * @param desc
   * @param data_id
   * @param name
   * @param config
   * @details Tensor必须为空
   */
  void create(Device *device, const TensorDesc &desc, int data_id,
              const std::string &name,
              const base::IntVector &config = base::IntVector());
  /**
   * @brief 创建一个tensor
   *
   * @param device
   * @param desc
   * @param data_id
   * @param config
   * @details Tensor必须为空
   */
  void create(Device *device, const TensorDesc &desc, int data_id,
              const base::IntVector &config = base::IntVector());
  /**
   * @brief 创建一个tensor
   *
   * @param desc
   * @param buffer
   * @param name
   * @details Tensor必须为空
   */
  void create(const TensorDesc &desc, Buffer *buffer, const std::string &name);
  /**
   * @brief 创建一个tensor
   *
   * @param desc
   * @param buffers
   * @details Tensor必须为空
   */
  void create(const TensorDesc &desc, Buffer *buffer);

  /**
   * @brief 让tensor变为空
   *
   */
  void destory();

  // alloc
  void allocBuffer(Device *device,
                   const base::IntVector &config = base::IntVector());
  void deallocateBuffer();

  // modify
  bool justModify(const TensorDesc &desc);
  bool justModify(Buffer *buffer);

  // get
  bool empty();
  bool isExternalBuffer();

  std::string getName();

  TensorDesc getDesc();
  base::DataType getDataType();
  base::DataFormat getDataFormat();
  base::IntVector getShape();
  int getShapeIndex(int index);
  int getBatch();
  int getChannel();
  int getDepth();
  int getHeight();
  int getWidth();
  base::SizeVector getStride();
  size_t getStrideIndex(int index);

  Buffer *getBuffer();
  base::DeviceType getDeviceType();
  Device *getDevice();
  BufferPool *getBufferPool();
  bool isBufferPool();
  BufferDesc getBufferDesc();
  size_t getSize();
  base::SizeVector getSizeVector();
  base::IntVector getConfig();
  void *getPtr();
  int getId();
  BufferSourceType getBufferSourceType();

 private:
  //! internal function
  void create(Device *device, const TensorDesc &desc, Buffer *buffer,
              void *data_ptr, int data_id, const std::string &name,
              const base::IntVector &config);
  void create(Device *device, const TensorDesc &desc, Buffer *buffer,
              void *data_ptr, int data_id, const base::IntVector &config);

 private:
  std::string name_ = "";            // tensor name
  TensorDesc desc_;                  // tensor desc
  bool is_external_buffer_ = false;  // 是否是外部buffer
  Buffer *buffer_ = nullptr;         // buffer
};

class TensorCreator {
 public:
  virtual ~TensorCreator(){};
  virtual Tensor *createTensor() = 0;
};

template <typename T>
class TypeTensorCreator : public TensorCreator {
  virtual Tensor *createTensor() { return new T(); }
};

std::map<base::TensorType, std::shared_ptr<TensorCreator>>
    &getGlobalTensorCreatorMap();

template <typename T>
class TypeTensorRegister {
 public:
  explicit TypeTensorRegister(base::TensorType type) {
    getGlobalTensorCreatorMap()[type] = std::shared_ptr<T>(new T());
  }
};

extern NNDEPLOY_CC_API Tensor *createTensor(base::TensorType type);

}  // namespace device
}  // namespace nndeploy

#endif
