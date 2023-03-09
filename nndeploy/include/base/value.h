/**
 * @file value.h
 * @author your name (you@domain.com)
 * @brief
 * @version 0.1
 * @date 2022-12-20
 *
 * @copyright Copyright (c) 2022
 * @note nndeploy c++的模板 trait等用法
 */
#ifndef _NNDEPLOY_INCLUDE_BASE_VALUE_H_
#define _NNDEPLOY_INCLUDE_BASE_VALUE_H_

#include "nndeploy/include/base/basic.h"
#include "nndeploy/include/base/include_c_cpp.h"
#include "nndeploy/include/base/macro.h"
#include "nndeploy/include/base/status.h"


namespace nndeploy {
namespace base {

/**
 * @brief
 * 不持有资源，所有资源都是外部管理的
 * 多种数据类型的构造函数
 * 能不能用模板
 * 返回函数可以是模板吗
 * 如何应对c++本身的强制类型转换函数呢
 * 运算符重载怎么搞呢
 */
class Value {
  Value();
  Value(const Value& value);
  Value(const Value&& value);

  Value(uint8_t value);
  Value(int8_t value);
  Value(uint16_t value);
  Value(int16_t value);
  Value(uint32_t value);
  Value(int32_t value);
  Value(uint64_t value);
  Value(int64_t value);
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
  Value(uint64_t* value, int64_t len);
  Value(int64_t* value, int64_t len);
  Value(float* value, int64_t len);
  Value(double* value, int64_t len);

  ~Value();

  bool operator=(const Value& value);
  bool operator=(const Value&& value);
  bool operator=(uint8_t value);
  bool operator=(int8_t value);
  bool operator=(uint16_t value);
  bool operator=(int16_t value);
  bool operator=(uint32_t value);
  bool operator=(int32_t value);
  bool operator=(uint64_t value);
  bool operator=(int64_t value);
  bool operator=(uint64_t value);
  bool operator=(int64_t value);
  bool operator=(float value);
  bool operator=(double value);

  void set(uint8_t value);
  void set(int8_t value);
  void set(uint16_t value);
  void set(int16_t value);
  void set(uint32_t value);
  void set(int32_t value);
  void set(uint64_t value);
  void set(int64_t value);
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
  bool get(uint64_t** value);
  bool get(int64_t** value);
  bool get(float** value);
  bool get(double** value);
  template <typename T>
  T get();
  template <typename T>
  T* get();

  bool operator==(const Value& value);
  bool operator==(uint8_t value);
  bool operator==(int8_t value);
  bool operator==(uint16_t value);
  bool operator==(int16_t value);
  bool operator==(uint32_t value);
  bool operator==(int32_t value);
  bool operator==(uint64_t value);
  bool operator==(int64_t value);
  bool operator==(uint64_t value);
  bool operator==(int64_t value);
  bool operator==(float value);
  bool operator==(double value);

  bool operator!=(const Value& value);
  bool operator!=(uint8_t value);
  bool operator!=(int8_t value);
  bool operator!=(uint16_t value);
  bool operator!=(int16_t value);
  bool operator!=(uint32_t value);
  bool operator!=(int32_t value);
  bool operator!=(uint64_t value);
  bool operator!=(int64_t value);
  bool operator!=(uint64_t value);
  bool operator!=(int64_t value);
  bool operator!=(float value);
  bool operator!=(double value);

  bool operator==(uint8_t* value);
  bool operator==(int8_t* value);
  bool operator==(uint16_t* value);
  bool operator==(int16_t* value);
  bool operator==(uint32_t* value);
  bool operator==(int32_t* value);
  bool operator==(uint64_t* value);
  bool operator==(int64_t* value);
  bool operator==(uint64_t* value);
  bool operator==(int64_t* value);
  bool operator==(float* value);
  bool operator==(double* value);

  bool operator!=(uint8_t* value);
  bool operator!=(int8_t* value);
  bool operator!=(uint16_t* value);
  bool operator!=(int16_t* value);
  bool operator!=(uint32_t* value);
  bool operator!=(int32_t* value);
  bool operator!=(uint64_t* value);
  bool operator!=(int64_t* value);
  bool operator!=(uint64_t* value);
  bool operator!=(int64_t* value);
  bool operator!=(float* value);
  bool operator!=(double* value);

  DataType getDataType();
  int64_t getLen();
  bool isValid();

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
  };

  DataType data_type_;
  int64_t len_ = -1;
  InternalValue internal_value_;
};

}  // namespace base
}  // namespace nndeploy

#endif