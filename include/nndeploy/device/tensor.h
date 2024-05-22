
#ifndef _NNDEPLOY_DEVICE_TENSOR_H_
#define _NNDEPLOY_DEVICE_TENSOR_H_

#include "nndeploy/base/common.h"
#include "nndeploy/base/glic_stl_include.h"
#include "nndeploy/base/log.h"
#include "nndeploy/base/macro.h"
#include "nndeploy/base/object.h"
#include "nndeploy/base/status.h"
#include "nndeploy/device/buffer.h"
#include "nndeploy/device/device.h"
#include "nndeploy/device/memory_pool.h"

namespace nndeploy {
namespace device {

/**
 * @brief Tensor类
 *
 */
class NNDEPLOY_CC_API Tensor {
 public:
  Tensor();
  Tensor(const std::string &name);
  Tensor(const TensorDesc &desc, const std::string &name = "");
  Tensor(const TensorDesc &desc, Buffer *buffer, const std::string &name = "");

  Tensor(Device *device, const TensorDesc &desc, const std::string &name = "",
         const base::IntVector &config = base::IntVector());
  Tensor(Device *device, const TensorDesc &desc, void *data_ptr,
         const std::string &name = "",
         const base::IntVector &config = base::IntVector());

  Tensor(MemoryPool *memory_pool, const TensorDesc &desc,
         const std::string &name = "",
         const base::IntVector &config = base::IntVector());
  Tensor(MemoryPool *memory_pool, const TensorDesc &desc, void *data_ptr,
         const std::string &name = "",
         const base::IntVector &config = base::IntVector());

  Tensor(const Tensor &tensor);
  Tensor &operator=(const Tensor &tensor);

  Tensor(Tensor &&tensor);
  Tensor &operator=(Tensor &&tensor);

  virtual ~Tensor();

  void create(const std::string &name);

  void create(const TensorDesc &desc, const std::string &name = "");
  void create(const TensorDesc &desc, Buffer *buffer,
              const std::string &name = "");

  void create(Device *device, const TensorDesc &desc,
              const std::string &name = "",
              const base::IntVector &config = base::IntVector());
  void create(Device *device, const TensorDesc &desc, void *data_ptr,
              const std::string &name = "",
              const base::IntVector &config = base::IntVector());

  void create(MemoryPool *memory_pool, const TensorDesc &desc,
              const std::string &name = "",
              const base::IntVector &config = base::IntVector());
  void create(MemoryPool *memory_pool, const TensorDesc &desc, void *data_ptr,
              const std::string &name = "",
              const base::IntVector &config = base::IntVector());

  // clear tensor
  void clear();

  // alloc
  void allocate(Device *device,
                const base::IntVector &config = base::IntVector());
  void allocate(MemoryPool *memory_pool,
                const base::IntVector &config = base::IntVector());
  void deallocate();

  // modify
  bool justModify(const TensorDesc &desc);
  bool justModify(Buffer *buffer);

  // clone and copy
  Tensor *clone();
  // dst必须预先分配内存
  base::Status copyTo(Tensor *dst);

  // bool
  bool isSameDevice(Tensor *tensor) const;
  bool isSameMemoryPool(Tensor *tensor) const;
  bool isSameDesc(Tensor *tensor) const;

  // get
  bool empty() const;
  bool isContinue() const;
  bool isExternalBuffer() const;

  std::string getName() const;
  TensorDesc getDesc() const;
  base::DataType getDataType() const;
  base::DataFormat getDataFormat() const;
  base::IntVector getShape() const;
  int getShapeIndex(int index) const;
  int getBatch() const;
  int getChannel() const;
  int getDepth() const;
  int getHeight() const;
  int getWidth() const;
  base::SizeVector getStride() const;
  size_t getStrideIndex(int index) const;

  Buffer *getBuffer() const;
  base::DeviceType getDeviceType() const;
  Device *getDevice() const;
  MemoryPool *getMemoryPool() const;
  bool isMemoryPool() const;
  BufferDesc getBufferDesc() const;
  size_t getSize() const;
  base::SizeVector getSizeVector() const;
  base::IntVector getConfig() const;
  void *getData() const;
  base::MemoryType getMemoryType() const;

  inline int addRef() const { return NNDEPLOY_XADD(ref_count_, 1); }
  inline int subRef() const { return NNDEPLOY_XADD(ref_count_, -1); }

 private:
  std::string name_ = "";     // tensor name
  TensorDesc desc_;           // tensor desc
  bool is_external_ = false;  // is external
  int *ref_count_;            // 引用计数
  Buffer *buffer_ = nullptr;  // buffer
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

std::map<base::TensorType, std::shared_ptr<TensorCreator>> &
getGlobalTensorCreatorMap();

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
