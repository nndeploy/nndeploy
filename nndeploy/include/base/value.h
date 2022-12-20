

struct Value {
union BasicDataType {
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

  void* ptr_void;
};

 DataType data_type;
 size_t len;

Status set(const T &value);
Status set(const T* value, size_t len);

Status get(T& value);
Status get(T* value, size_t &len);

}