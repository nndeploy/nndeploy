#ifndef _NNDEPLOY_DEVICE_TYPE_H_
#define _NNDEPLOY_DEVICE_TYPE_H_

#include "nndeploy/base/common.h"
#include "nndeploy/base/glic_stl_include.h"
#include "nndeploy/base/log.h"
#include "nndeploy/base/macro.h"
#include "nndeploy/base/object.h"
#include "nndeploy/base/status.h"

namespace nndeploy {
namespace device {

/**
 * @brief TensorDesc
 *
 */
struct NNDEPLOY_CC_API BufferDesc {
  BufferDesc();
  explicit BufferDesc(size_t size);
  explicit BufferDesc(size_t *size, size_t len);
  explicit BufferDesc(const base::SizeVector &size);
  explicit BufferDesc(size_t size, const base::IntVector &config);
  explicit BufferDesc(const base::SizeVector &size,
                      const base::IntVector &config);
  explicit BufferDesc(size_t *size, size_t len, const base::IntVector &config);

  BufferDesc(const BufferDesc &desc);
  BufferDesc &operator=(const BufferDesc &desc);
  BufferDesc &operator=(size_t size);

  BufferDesc(BufferDesc &&desc) noexcept;
  BufferDesc &operator=(BufferDesc &&desc) noexcept;

  virtual ~BufferDesc();

  size_t getSize() const;
  base::SizeVector getSizeVector() const;
  size_t getRealSize() const;
  base::SizeVector getRealSizeVector() const;

  base::IntVector getConfig() const;

  bool isSameConfig(const BufferDesc &desc) const;
  bool isSameDim(const BufferDesc &desc) const;
  bool is1D() const;

  bool operator>=(const BufferDesc &other) const;
  bool operator==(const BufferDesc &other) const;
  bool operator!=(const BufferDesc &other) const;

  // base::Status serialize(std::string &bin_str);
  // base::Status deserialize(const std::string &bin_str);

  void print(std::ostream &stream = std::cout) const;

  bool justModify(const size_t &size);
  bool justModify(const base::SizeVector &size);
  bool justModify(const BufferDesc &desc);

  void clear();

 private:
  /**
   * @brief
   * 1d size
   * 2d h w c sizeof(T) - 例如OpenCL cl::Image2d
   * 3d unknown
   */
  base::SizeVector size_;
  /**
   * @brief
   * 根据不同的设备以及内存形态有不同的config_
   */
  base::IntVector config_;
  /**
   * @brief
   * 真实内存大小，无法修改
   */
  base::SizeVector real_size_;
};

/**
 * @brief TensorDesc
 *
 */
struct NNDEPLOY_CC_API TensorDesc {
  TensorDesc();
  explicit TensorDesc(base::DataType data_type, base::DataFormat format,
                      const base::IntVector &shape);
  explicit TensorDesc(base::DataType data_type, base::DataFormat format,
                      const base::IntVector &shape,
                      const base::SizeVector &stride);

  TensorDesc(const TensorDesc &desc);
  TensorDesc &operator=(const TensorDesc &desc);

  TensorDesc(TensorDesc &&desc) noexcept;
  TensorDesc &operator=(TensorDesc &&desc) noexcept;

  virtual ~TensorDesc();

  bool operator==(const TensorDesc &other) const;
  bool operator!=(const TensorDesc &other) const;

  base::Status serialize(std::string &bin_str);
  base::Status deserialize(const std::string &bin_str);

  void print(std::ostream &stream = std::cout) const;

  base::DataType data_type_ = base::dataTypeOf<float>();        // 数据类型
  base::DataFormat data_format_ = base::kDataFormatNotSupport;  // 数据格式
  base::IntVector shape_;                                       // 数据形状
  base::SizeVector stride_;                                     // 数据步长
};

}  // namespace device
}  // namespace nndeploy

#endif /* _NNDEPLOY_DEVICE_TYPE_H_ */
