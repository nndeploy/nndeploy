
#ifndef _NNDEPLOY_BASE_VALUE_H_
#define _NNDEPLOY_BASE_VALUE_H_

#include "nndeploy/base/common.h"
#include "nndeploy/base/glic_stl_include.h"
#include "nndeploy/base/macro.h"
#include "nndeploy/base/status.h"

namespace nndeploy {
namespace base {

class NNDEPLOY_CC_API Value {
 public:
  Value() = default;
  Value(const Value& value) = default;
  Value(Value&& value) noexcept {
    data_type_ = value.data_type_;
    len_ = value.len_;
    internal_value_ = value.internal_value_;
  }

  template <typename T>
  Value(T value) {
    set(value);
  }

  template <typename T>
  Value(T* value, int64_t len) {
    set(value, len);
  }

  ~Value(){};

  Value& operator=(const Value& value) = default;
  Value& operator=(Value&& value) noexcept {
    data_type_ = value.data_type_;
    len_ = value.len_;
    internal_value_ = value.internal_value_;
    return *this;
  }

  template <typename T>
  void set(T value) {
    data_type_ = dataTypeOf<T>();
    len_ = 0;
    internal_value_ = (void*)(&value);
  }

  template <typename T>
  void set(T* value, int64_t len) {
    data_type_ = dataTypeOf<T>();
    len_ = 0;
    internal_value_ = (void*)(&value);
  }

  template <typename T>
  T get() {
    DataType data_type = dataTypeOf<T>();
    if (data_type_ == data_type && len_ == 0) {
      return (*(T*)(internal_value_));
    } else {
      return T();
    }
  }

  template <typename T>
  T* get(int64_t& len) {
    DataType data_type = dataTypeOf<T>();
    if (data_type_ == data_type && len_ >= 0) {
      len = len_;
      return (T*)(internal_value_);
    } else {
      return nullptr;
    }
  }

  DataType getDataType() const { return data_type_; }
  int64_t getLen() const { return len_; }
  bool isValid() const { return len_ >= 0; }

 private:
  DataType data_type_;
  /**
   * @brief len_ is the length of the data
   * if the data is a pointer, len_ >= 1
   * if the data is a scalar, len_ == 0
   * if the data is invalid, len_ == -1
   */
  int64_t len_ = -1;
  void* internal_value_;
};

}  // namespace base
}  // namespace nndeploy

#endif