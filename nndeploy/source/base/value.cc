#include "nndeploy/include/base/value.h"

namespace nndeploy {
namespace base {

/**
 * @brief 不持有资源，所有资源都是外部管理的
 * 多种数据类型的构造函数
 *
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

  operator=(const Value& value);
  operator=(const Value&& value);
  
  operator=(uint8_t value);
  operator=(int8_t value);
  operator=(uint16_t value);
  operator=(int16_t value);
  operator=(uint32_t value);
  operator=(int32_t value);
  operator=(uint64_t value);
  operator=(int64_t value);
  operator=(uint64_t value);
  operator=(int64_t value);
  operator=(float value);
  operator=(double value);

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

  uint8_t get();
  int8_t get();
  uint16_t get();
  int16_t get();
  uint32_t get();
  int32_t get();
  uint64_t get();
  int64_t get();
  uint64_t get();
  int64_t get();
  float get();
  double get();

  uint8_t* get();
  int8_t* get();
  uint16_t* get();
  int16_t* get();
  uint32_t* get();
  int32_t* get();
  uint64_t* get();
  int64_t* get();
  uint64_t* get();
  int64_t* get();
  float* get();
  double* get();

  operator int32_t();

  bool operator==(int32_t value);
  bool operator!=(int32_t value);

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
  InternalValue value_;
};

}  // namespace base
}  // namespace nndeploy
