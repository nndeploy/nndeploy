
#include "nndeploy/base/basic.h"

namespace nndeploy {
namespace base {

template <>
DataType dataTypeOf<float>() {
  return DataType(kDataTypeCodeFp, 32);
}

template <>
DataType dataTypeOf<double>() {
  return DataType(kDataTypeCodeFp, 64);
}

template <>
DataType dataTypeOf<uint8_t>() {
  return DataType(kDataTypeCodeUint, 8);
}

template <>
DataType dataTypeOf<uint16_t>() {
  return DataType(kDataTypeCodeUint, 16);
}

template <>
DataType dataTypeOf<uint32_t>() {
  return DataType(kDataTypeCodeUint, 32);
}

template <>
DataType dataTypeOf<uint64_t>() {
  return DataType(kDataTypeCodeUint, 64);
}

template <>
DataType dataTypeOf<int8_t>() {
  return DataType(kDataTypeCodeInt, 8);
}

template <>
DataType dataTypeOf<int16_t>() {
  return DataType(kDataTypeCodeInt, 16);
}

template <>
DataType dataTypeOf<int32_t>() {
  return DataType(kDataTypeCodeInt, 32);
}

template <>
DataType dataTypeOf<int64_t>() {
  return DataType(kDataTypeCodeInt, 64);
}

}  // namespace base
}  // namespace nndeploy