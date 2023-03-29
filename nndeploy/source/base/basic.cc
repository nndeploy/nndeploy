
#include "nndeploy/source/base/basic.h"

namespace nndeploy {
namespace base {

template <>
DataType DataTypeOf<float>() {
  return DataType(kDataTypeCodeFp, 32);
}

template <>
DataType DataTypeOf<double>() {
  return DataType(kDataTypeCodeFp, 64);
}

template <>
DataType DataTypeOf<uint8_t>() {
  return DataType(kDataTypeCodeUint, 8);
}

template <>
DataType DataTypeOf<uint16_t>() {
  return DataType(kDataTypeCodeUint, 16);
}

template <>
DataType DataTypeOf<uint32_t>() {
  return DataType(kDataTypeCodeUint, 32);
}

template <>
DataType DataTypeOf<uint64_t>() {
  return DataType(kDataTypeCodeUint, 64);
}

template <>
DataType DataTypeOf<int8_t>() {
  return DataType(kDataTypeCodeInt, 8);
}

template <>
DataType DataTypeOf<int16_t>() {
  return DataType(kDataTypeCodeInt, 16);
}

template <>
DataType DataTypeOf<int32_t>() {
  return DataType(kDataTypeCodeInt, 32);
}

template <>
DataType DataTypeOf<int64_t>() {
  return DataType(kDataTypeCodeInt, 64);
}

}  // namespace base
}  // namespace nndeploy