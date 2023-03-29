
#ifndef _NNDEPLOY_SOURCE_BASE_VALUE_H_
#define _NNDEPLOY_SOURCE_BASE_VALUE_H_

#include "nndeploy/source/base/basic.h"
#include "nndeploy/source/base/include_c_cpp.h"
#include "nndeploy/source/base/macro.h"
#include "nndeploy/source/base/status.h"

namespace nndeploy {
namespace base {

// 增加type_trait
class NNDEPLOY_CC_API Value {
 public:
  Value() = default;
  Value(const Value& value) = default;
  Value(Value&& value) noexcept {
    data_type_ = value.data_type_;
    len_ = value.len_;
    internal_value_ = value.internal_value_;
  };

  Value(uint8_t value);
  Value(int8_t value);
  Value(uint16_t value);
  Value(int16_t value);
  Value(uint32_t value);
  Value(int32_t value);
  Value(uint64_t value);
  Value(int64_t value);
  Value(float value);
  Value(double value);

  Value(uint8_t* value, int64_t len);
  Value(int8_t* value, int64_t len);
  Value(uint16_t* value, int64_t len);
  Value(int16_t* value, int64_t len);
  Value(uint32_t* value, int64_t len);
  Value(int32_t* value, int64_t len);
  Value(uint64_t* value, int64_t len);
  Value(int64_t* value, int64_t len);
  Value(float* value, int64_t len);
  Value(double* value, int64_t len);

  ~Value(){};

  Value& operator=(const Value& value) = default;
  Value& operator=(Value&& value) noexcept {
    data_type_ = value.data_type_;
    len_ = value.len_;
    internal_value_ = value.internal_value_;
    return *this;
  }

  Value& operator=(uint8_t value);
  Value& operator=(int8_t value);
  Value& operator=(uint16_t value);
  Value& operator=(int16_t value);
  Value& operator=(uint32_t value);
  Value& operator=(int32_t value);
  Value& operator=(uint64_t value);
  Value& operator=(int64_t value);
  Value& operator=(float value);
  Value& operator=(double value);

  void set(uint8_t value);
  void set(int8_t value);
  void set(uint16_t value);
  void set(int16_t value);
  void set(uint32_t value);
  void set(int32_t value);
  void set(uint64_t value);
  void set(int64_t value);
  void set(float value);
  void set(double value);

  void set(uint8_t* value, int64_t len);
  void set(int8_t* value, int64_t len);
  void set(uint16_t* value, int64_t len);
  void set(int16_t* value, int64_t len);
  void set(uint32_t* value, int64_t len);
  void set(int32_t* value, int64_t len);
  void set(uint64_t* value, int64_t len);
  void set(int64_t* value, int64_t len);
  void set(float* value, int64_t len);
  void set(double* value, int64_t len);

  bool get(uint8_t& value);
  bool get(int8_t& value);
  bool get(uint16_t& value);
  bool get(int16_t& value);
  bool get(uint32_t& value);
  bool get(int32_t& value);
  bool get(uint64_t& value);
  bool get(int64_t& value);
  bool get(float& value);
  bool get(double& value);

  bool get(uint8_t** value);
  bool get(int8_t** value);
  bool get(uint16_t** value);
  bool get(int16_t** value);
  bool get(uint32_t** value);
  bool get(int32_t** value);
  bool get(uint64_t** value);
  bool get(int64_t** value);
  bool get(float** value);
  bool get(double** value);

  DataType getDataType() const;
  int64_t getLen() const;
  bool isValid() const;

 private:
  union InternalValue {
    uint8_t value_u8;
    int8_t value_i8;
    uint16_t value_u16;
    int16_t value_i16;
    uint32_t value_u32;
    int32_t value_i32;
    uint64_t value_u64;
    int64_t value_i64;
    size_t value_size;
    float value_f32;
    double value_f64;

    uint8_t* ptr_u8;
    int8_t* ptr_i8;
    uint16_t* ptr_u16;
    int16_t* ptr_i16;
    uint32_t* ptr_u32;
    int32_t* ptr_i32;
    uint64_t* ptr_u64;
    int64_t* ptr_i64;
    size_t* ptr_size;
    float* ptr_f32;
    double* ptr_f64;
    /**
     * @brief ptr_void is used to store the pointer of any type
     * data_type_.code_ = kDataTypeCodeOpaqueHandle
     * data_type_.bits_ = 1
     * data_type_.lanes_ = 1
     * len_ = len * sizeof(any type)
     * is weird but useful
     */
    void* ptr_void;
  };

  DataType data_type_;
  /**
   * @brief len_ is the length of the data
   * if the data is a pointer, len_ >= 1
   * if the data is a scalar, len_ == 0
   * if the data is invalid, len_ == -1
   */
  int64_t len_ = -1;
  InternalValue internal_value_;
};

}  // namespace base
}  // namespace nndeploy

#endif