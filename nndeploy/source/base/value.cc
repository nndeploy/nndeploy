
#include "nndeploy/source/base/value.h"

namespace nndeploy {
namespace base {

Value::Value(uint8_t value) {
  data_type_.code_ = kDataTypeCodeUint;
  data_type_.bits_ = 1;
  data_type_.lanes_ = 1;
  len_ = 0;
  internal_value_.value_u8 = value;
}
Value::Value(int8_t value) {
  data_type_.code_ = kDataTypeCodeInt;
  data_type_.bits_ = 1;
  data_type_.lanes_ = 1;
  len_ = 0;
  internal_value_.value_i8 = value;
}
Value::Value(uint16_t value) {
  data_type_.code_ = kDataTypeCodeUint;
  data_type_.bits_ = 2;
  data_type_.lanes_ = 1;
  len_ = 0;
  internal_value_.value_u16 = value;
}
Value::Value(int16_t value) {
  data_type_.code_ = kDataTypeCodeInt;
  data_type_.bits_ = 2;
  data_type_.lanes_ = 1;
  len_ = 0;
  internal_value_.value_i16 = value;
}
Value::Value(uint32_t value) {
  data_type_.code_ = kDataTypeCodeUint;
  data_type_.bits_ = 4;
  data_type_.lanes_ = 1;
  len_ = 0;
  internal_value_.value_u32 = value;
}
Value::Value(int32_t value) {
  data_type_.code_ = kDataTypeCodeInt;
  data_type_.bits_ = 4;
  data_type_.lanes_ = 1;
  len_ = 0;
  internal_value_.value_i32 = value;
}
Value::Value(uint64_t value) {
  data_type_.code_ = kDataTypeCodeUint;
  data_type_.bits_ = 8;
  data_type_.lanes_ = 1;
  len_ = 0;
  internal_value_.value_u64 = value;
}
Value::Value(int64_t value) {
  data_type_.code_ = kDataTypeCodeInt;
  data_type_.bits_ = 8;
  data_type_.lanes_ = 1;
  len_ = 0;
  internal_value_.value_i64 = value;
}
Value::Value(float value) {
  data_type_.code_ = kDataTypeCodeFp;
  data_type_.bits_ = 4;
  data_type_.lanes_ = 1;
  len_ = 0;
  internal_value_.value_f32 = value;
}
Value::Value(double value) {
  data_type_.code_ = kDataTypeCodeFp;
  data_type_.bits_ = 8;
  data_type_.lanes_ = 1;
  len_ = 0;
  internal_value_.value_f64 = value;
}

Value::Value(uint8_t* value, int64_t len) {
  data_type_.code_ = kDataTypeCodeUint;
  data_type_.bits_ = 1;
  data_type_.lanes_ = 1;
  len_ = len;
  internal_value_.ptr_u8 = value;
}
Value::Value(int8_t* value, int64_t len) {
  data_type_.code_ = kDataTypeCodeInt;
  data_type_.bits_ = 1;
  data_type_.lanes_ = 1;
  len_ = len;
  internal_value_.ptr_i8 = value;
}
Value::Value(uint16_t* value, int64_t len) {
  data_type_.code_ = kDataTypeCodeUint;
  data_type_.bits_ = 2;
  data_type_.lanes_ = 1;
  len_ = len;
  internal_value_.ptr_u16 = value;
}
Value::Value(int16_t* value, int64_t len) {
  data_type_.code_ = kDataTypeCodeInt;
  data_type_.bits_ = 2;
  data_type_.lanes_ = 1;
  len_ = len;
  internal_value_.ptr_i16 = value;
}
Value::Value(uint32_t* value, int64_t len) {
  data_type_.code_ = kDataTypeCodeUint;
  data_type_.bits_ = 4;
  data_type_.lanes_ = 1;
  len_ = len;
  internal_value_.ptr_u32 = value;
}
Value::Value(int32_t* value, int64_t len) {
  data_type_.code_ = kDataTypeCodeInt;
  data_type_.bits_ = 4;
  data_type_.lanes_ = 1;
  len_ = len;
  internal_value_.ptr_i32 = value;
}
Value::Value(uint64_t* value, int64_t len) {
  data_type_.code_ = kDataTypeCodeUint;
  data_type_.bits_ = 8;
  data_type_.lanes_ = 1;
  len_ = len;
  internal_value_.ptr_u64 = value;
}
Value::Value(int64_t* value, int64_t len) {
  data_type_.code_ = kDataTypeCodeInt;
  data_type_.bits_ = 8;
  data_type_.lanes_ = 1;
  len_ = len;
  internal_value_.ptr_i64 = value;
}
Value::Value(float* value, int64_t len) {
  data_type_.code_ = kDataTypeCodeFp;
  data_type_.bits_ = 4;
  data_type_.lanes_ = 1;
  len_ = len;
  internal_value_.ptr_f32 = value;
}
Value::Value(double* value, int64_t len) {
  data_type_.code_ = kDataTypeCodeFp;
  data_type_.bits_ = 8;
  data_type_.lanes_ = 1;
  len_ = len;
  internal_value_.ptr_f64 = value;
}

Value& Value::operator=(uint8_t value) {
  data_type_.code_ = kDataTypeCodeUint;
  data_type_.bits_ = 1;
  data_type_.lanes_ = 1;
  len_ = 0;
  internal_value_.value_u8 = value;
  return *this;
}
Value& Value::operator=(int8_t value) {
  data_type_.code_ = kDataTypeCodeInt;
  data_type_.bits_ = 1;
  data_type_.lanes_ = 1;
  len_ = 0;
  internal_value_.value_i8 = value;
  return *this;
}
Value& Value::operator=(uint16_t value) {
  data_type_.code_ = kDataTypeCodeUint;
  data_type_.bits_ = 2;
  data_type_.lanes_ = 1;
  len_ = 0;
  internal_value_.value_u16 = value;
  return *this;
}
Value& Value::operator=(int16_t value) {
  data_type_.code_ = kDataTypeCodeInt;
  data_type_.bits_ = 2;
  data_type_.lanes_ = 1;
  len_ = 0;
  internal_value_.value_i16 = value;
  return *this;
}
Value& Value::operator=(uint32_t value) {
  data_type_.code_ = kDataTypeCodeUint;
  data_type_.bits_ = 4;
  data_type_.lanes_ = 1;
  len_ = 0;
  internal_value_.value_u32 = value;
  return *this;
}
Value& Value::operator=(int32_t value) {
  data_type_.code_ = kDataTypeCodeInt;
  data_type_.bits_ = 4;
  data_type_.lanes_ = 1;
  len_ = 0;
  internal_value_.value_i32 = value;
  return *this;
}
Value& Value::operator=(uint64_t value) {
  data_type_.code_ = kDataTypeCodeUint;
  data_type_.bits_ = 8;
  data_type_.lanes_ = 1;
  len_ = 0;
  internal_value_.value_u64 = value;
  return *this;
}
Value& Value::operator=(int64_t value) {
  data_type_.code_ = kDataTypeCodeInt;
  data_type_.bits_ = 8;
  data_type_.lanes_ = 1;
  len_ = 0;
  internal_value_.value_i64 = value;
  return *this;
}
Value& Value::operator=(float value) {
  data_type_.code_ = kDataTypeCodeFp;
  data_type_.bits_ = 4;
  data_type_.lanes_ = 1;
  len_ = 0;
  internal_value_.value_f32 = value;
  return *this;
}
Value& Value::operator=(double value) {
  data_type_.code_ = kDataTypeCodeFp;
  data_type_.bits_ = 8;
  data_type_.lanes_ = 1;
  len_ = 0;
  internal_value_.value_f64 = value;
  return *this;
}

void Value::set(uint8_t value) {
  data_type_.code_ = kDataTypeCodeUint;
  data_type_.bits_ = 1;
  data_type_.lanes_ = 1;
  len_ = 0;
  internal_value_.value_u8 = value;
}
void Value::set(int8_t value) {
  data_type_.code_ = kDataTypeCodeInt;
  data_type_.bits_ = 1;
  data_type_.lanes_ = 1;
  len_ = 0;
  internal_value_.value_i8 = value;
}
void Value::set(uint16_t value) {
  data_type_.code_ = kDataTypeCodeUint;
  data_type_.bits_ = 2;
  data_type_.lanes_ = 1;
  len_ = 0;
  internal_value_.value_u16 = value;
}
void Value::set(int16_t value) {
  data_type_.code_ = kDataTypeCodeInt;
  data_type_.bits_ = 2;
  data_type_.lanes_ = 1;
  len_ = 0;
  internal_value_.value_i16 = value;
}
void Value::set(uint32_t value) {
  data_type_.code_ = kDataTypeCodeUint;
  data_type_.bits_ = 4;
  data_type_.lanes_ = 1;
  len_ = 0;
  internal_value_.value_u32 = value;
}
void Value::set(int32_t value) {
  data_type_.code_ = kDataTypeCodeInt;
  data_type_.bits_ = 4;
  data_type_.lanes_ = 1;
  len_ = 0;
  internal_value_.value_i32 = value;
}
void Value::set(uint64_t value) {
  data_type_.code_ = kDataTypeCodeUint;
  data_type_.bits_ = 8;
  data_type_.lanes_ = 1;
  len_ = 0;
  internal_value_.value_u64 = value;
}
void Value::set(int64_t value) {
  data_type_.code_ = kDataTypeCodeInt;
  data_type_.bits_ = 8;
  data_type_.lanes_ = 1;
  len_ = 0;
  internal_value_.value_i64 = value;
}
void Value::set(float value) {
  data_type_.code_ = kDataTypeCodeFp;
  data_type_.bits_ = 4;
  data_type_.lanes_ = 1;
  len_ = 0;
  internal_value_.value_f32 = value;
}
void Value::set(double value) {
  data_type_.code_ = kDataTypeCodeFp;
  data_type_.bits_ = 8;
  data_type_.lanes_ = 1;
  len_ = 0;
  internal_value_.value_f64 = value;
}

void Value::set(uint8_t* value, int64_t len) {
  data_type_.code_ = kDataTypeCodeUint;
  data_type_.bits_ = 1;
  data_type_.lanes_ = 1;
  len_ = len;
  internal_value_.ptr_u8 = value;
}
void Value::set(int8_t* value, int64_t len) {
  data_type_.code_ = kDataTypeCodeInt;
  data_type_.bits_ = 1;
  data_type_.lanes_ = 1;
  len_ = len;
  internal_value_.ptr_i8 = value;
}
void Value::set(uint16_t* value, int64_t len) {
  data_type_.code_ = kDataTypeCodeUint;
  data_type_.bits_ = 2;
  data_type_.lanes_ = 1;
  len_ = len;
  internal_value_.ptr_u16 = value;
}
void Value::set(int16_t* value, int64_t len) {
  data_type_.code_ = kDataTypeCodeInt;
  data_type_.bits_ = 2;
  data_type_.lanes_ = 1;
  len_ = len;
  internal_value_.ptr_i16 = value;
}
void Value::set(uint32_t* value, int64_t len) {
  data_type_.code_ = kDataTypeCodeUint;
  data_type_.bits_ = 4;
  data_type_.lanes_ = 1;
  len_ = len;
  internal_value_.ptr_u32 = value;
}
void Value::set(int32_t* value, int64_t len) {
  data_type_.code_ = kDataTypeCodeInt;
  data_type_.bits_ = 4;
  data_type_.lanes_ = 1;
  len_ = len;
  internal_value_.ptr_i32 = value;
}
void Value::set(uint64_t* value, int64_t len) {
  data_type_.code_ = kDataTypeCodeUint;
  data_type_.bits_ = 8;
  data_type_.lanes_ = 1;
  len_ = len;
  internal_value_.ptr_u64 = value;
}
void Value::set(int64_t* value, int64_t len) {
  data_type_.code_ = kDataTypeCodeInt;
  data_type_.bits_ = 8;
  data_type_.lanes_ = 1;
  len_ = len;
  internal_value_.ptr_i64 = value;
}
void Value::set(float* value, int64_t len) {
  data_type_.code_ = kDataTypeCodeFp;
  data_type_.bits_ = 4;
  data_type_.lanes_ = 1;
  len_ = len;
  internal_value_.ptr_f32 = value;
}
void Value::set(double* value, int64_t len) {
  data_type_.code_ = kDataTypeCodeFp;
  data_type_.bits_ = 8;
  data_type_.lanes_ = 1;
  len_ = len;
  internal_value_.ptr_f64 = value;
}

bool Value::get(uint8_t& value) {
  if (data_type_.code_ != kDataTypeCodeUint || data_type_.bits_ != 1 ||
      data_type_.lanes_ != 1 || len_ != 0) {
    return false;
  }
  value = internal_value_.value_u8;
  return true;
}
bool Value::get(int8_t& value) {
  if (data_type_.code_ != kDataTypeCodeInt || data_type_.bits_ != 1 ||
      data_type_.lanes_ != 1 || len_ != 0) {
    return false;
  }
  value = internal_value_.value_i8;
  return true;
}
bool Value::get(uint16_t& value) {
  if (data_type_.code_ != kDataTypeCodeUint || data_type_.bits_ != 2 ||
      data_type_.lanes_ != 1 || len_ != 0) {
    return false;
  }
  value = internal_value_.value_u16;
  return true;
}
bool Value::get(int16_t& value) {
  if (data_type_.code_ != kDataTypeCodeInt || data_type_.bits_ != 2 ||
      data_type_.lanes_ != 1 || len_ != 0) {
    return false;
  }
  value = internal_value_.value_i16;
  return true;
}
bool Value::get(uint32_t& value) {
  if (data_type_.code_ != kDataTypeCodeUint || data_type_.bits_ != 4 ||
      data_type_.lanes_ != 1 || len_ != 0) {
    return false;
  }
  value = internal_value_.value_u32;
  return true;
}
bool Value::get(int32_t& value) {
  if (data_type_.code_ != kDataTypeCodeInt || data_type_.bits_ != 4 ||
      data_type_.lanes_ != 1 || len_ != 0) {
    return false;
  }
  value = internal_value_.value_i32;
  return true;
}
bool Value::get(uint64_t& value) {
  if (data_type_.code_ != kDataTypeCodeUint || data_type_.bits_ != 8 ||
      data_type_.lanes_ != 1 || len_ != 0) {
    return false;
  }
  value = internal_value_.value_u64;
  return true;
}
bool Value::get(int64_t& value) {
  if (data_type_.code_ != kDataTypeCodeInt || data_type_.bits_ != 8 ||
      data_type_.lanes_ != 1 || len_ != 0) {
    return false;
  }
  value = internal_value_.value_i64;
  return true;
}
bool Value::get(float& value) {
  if (data_type_.code_ != kDataTypeCodeFp || data_type_.bits_ != 4 ||
      data_type_.lanes_ != 1 || len_ != 0) {
    return false;
  }
  value = internal_value_.value_f32;
  return true;
}
bool Value::get(double& value) {
  if (data_type_.code_ != kDataTypeCodeFp || data_type_.bits_ != 8 ||
      data_type_.lanes_ != 1 || len_ != 0) {
    return false;
  }
  value = internal_value_.value_f64;
  return true;
}

bool Value::get(uint8_t** value) {
  if (data_type_.code_ != kDataTypeCodeUint || data_type_.bits_ != 1 ||
      data_type_.lanes_ != 1 || len_ < 1) {
    return false;
  }
  *value = internal_value_.ptr_u8;
  return true;
}
bool Value::get(int8_t** value) {
  if (data_type_.code_ != kDataTypeCodeInt || data_type_.bits_ != 1 ||
      data_type_.lanes_ != 1 || len_ < 1) {
    return false;
  }
  *value = internal_value_.ptr_i8;
  return true;
}
bool Value::get(uint16_t** value) {
  if (data_type_.code_ != kDataTypeCodeUint || data_type_.bits_ != 2 ||
      data_type_.lanes_ != 1 || len_ < 1) {
    return false;
  }
  *value = internal_value_.ptr_u16;
  return true;
}
bool Value::get(int16_t** value) {
  if (data_type_.code_ != kDataTypeCodeInt || data_type_.bits_ != 2 ||
      data_type_.lanes_ != 1 || len_ < 1) {
    return false;
  }
  *value = internal_value_.ptr_i16;
  return true;
}
bool Value::get(uint32_t** value) {
  if (data_type_.code_ != kDataTypeCodeUint || data_type_.bits_ != 4 ||
      data_type_.lanes_ != 1 || len_ < 1) {
    return false;
  }
  *value = internal_value_.ptr_u32;
  return true;
}
bool Value::get(int32_t** value) {
  if (data_type_.code_ != kDataTypeCodeInt || data_type_.bits_ != 4 ||
      data_type_.lanes_ != 1 || len_ < 1) {
    return false;
  }
  *value = internal_value_.ptr_i32;
  return true;
}
bool Value::get(uint64_t** value) {
  if (data_type_.code_ != kDataTypeCodeUint || data_type_.bits_ != 8 ||
      data_type_.lanes_ != 1 || len_ < 1) {
    return false;
  }
  *value = internal_value_.ptr_u64;
  return true;
}
bool Value::get(int64_t** value) {
  if (data_type_.code_ != kDataTypeCodeInt || data_type_.bits_ != 8 ||
      data_type_.lanes_ != 1 || len_ < 1) {
    return false;
  }
  *value = internal_value_.ptr_i64;
  return true;
}
bool Value::get(float** value) {
  if (data_type_.code_ != kDataTypeCodeFp || data_type_.bits_ != 4 ||
      data_type_.lanes_ != 1 || len_ < 1) {
    return false;
  }
  *value = internal_value_.ptr_f32;
  return true;
}
bool Value::get(double** value) {
  if (data_type_.code_ != kDataTypeCodeFp || data_type_.bits_ != 8 ||
      data_type_.lanes_ != 1 || len_ < 1) {
    return false;
  }
  *value = internal_value_.ptr_f64;
  return true;
}

DataType Value::getDataType() const { return data_type_; };
int64_t Value::getLen() const { return len_; };
bool Value::isValid() const { return len_ >= 0; };

}  // namespace base
}  // namespace nndeploy
