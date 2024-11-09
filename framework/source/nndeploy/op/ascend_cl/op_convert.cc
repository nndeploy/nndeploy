
#include "nndeploy/op/ascend_cl/op_convert.h"

#include "nndeploy/base/common.h"
#include "nndeploy/base/half.h"
#include "nndeploy/base/half.hpp"
#include "nndeploy/base/log.h"
#include "nndeploy/base/macro.h"
#include "nndeploy/base/status.h"
#include "nndeploy/base/string.h"
#include "nndeploy/base/type.h"
#include "nndeploy/device/device.h"
#include "nndeploy/device/tensor.h"
#include "nndeploy/op/ascend_cl/op_include.h"

namespace nndeploy {
namespace op {

constexpr int BLOCKSIZE = 16;

// base format is ND/NCHW
base::IntVector inferShapeLessTo4(base::IntVector dims) {
  base::IntVector res;
  res.resize(4);
  if (dims.size() > 4) {
    NNDEPLOY_LOGE("input dim > 4 when inferShapeLessTo4.\n");
    return base::IntVector();
  }
  switch (dims.size()) {
    case 0:  // 标量
      res[0] = 1;
      res[1] = 1;
      res[2] = 1;
      res[3] = 1;
      break;
    case 1:  // kDataFormatN、kDataFormatNC
      res[0] = 1;
      res[1] = dims[0];
      res[2] = 1;
      res[3] = 1;
      break;
    case 2:
      res[0] = 1;
      res[1] = dims[0];
      res[2] = dims[1];
      res[3] = 1;
      break;
    case 3:
      res[0] = 1;
      res[1] = dims[0];
      res[2] = dims[1];
      res[3] = dims[2];
      break;
    case 4:
      res[0] = dims[0];
      res[1] = dims[1];
      res[2] = dims[2];
      res[3] = dims[3];
      break;
    default:
      NNDEPLOY_LOGE("input dim is error when inferShapeLessTo4.\n");
      res.clear();
  }
  return res;
}

base::IntVector inferShape4To5(base::IntVector dims) {
  base::IntVector res;
  res.resize(5);
  if (dims.size() < 4) {
    return inferShape4To5(inferShapeLessTo4(dims));
  } else if (dims.size() > 4) {
    NNDEPLOY_LOGE("infershape4to5 but input dim > 4.\n");
  }
  res[0] = dims[0];
  res[1] = (dims[1] + 15) / 16;
  res[2] = dims[2];
  res[3] = dims[3];
  res[4] = BLOCKSIZE;
  return res;
}

base::IntVector inferShape5To4(base::IntVector dims) {
  base::IntVector res;
  res.emplace_back(dims[0]);
  res.emplace_back(((dims[1] + 15) / 16) * 16);
  res.emplace_back(dims[2]);
  res.emplace_back(dims[3]);
  return res;
}

base::IntVector inferShapeNDToNZ(base::IntVector dims) {
  base::IntVector res;
  // sum(keepdim = false) may make tensor dim = 0
  base::IntVector dim;
  for (int i = 0; i < dims.size(); i++) {
    dim.emplace_back(dims[i]);
  }

  // TODO(ascend): this expand code can be remove now
  // this action will move to GuessStorageSizeWhenConvertFormat
  if (dim.size() == 0) {
    dim.emplace_back(1);
  }
  if (dim.size() == 1) {
    dim.emplace_back(1);
  }

  int i = 0;
  for (; i < dim.size() - 2; i++) {
    res.emplace_back(dim[i]);
  }

  res.emplace_back((dim[i + 1] + 15) / BLOCKSIZE);
  res.emplace_back((dim[i] + 15) / BLOCKSIZE);
  res.emplace_back(BLOCKSIZE);
  res.emplace_back(BLOCKSIZE);

  return res;
}

base::IntVector inferShapeNDToZ(base::IntVector dims) {
  base::IntVector res;
  if (dims.size() < 4) {
    return inferShapeNDToZ(inferShapeLessTo4(dims));
  }

  res.emplace_back((dims[1] + 15) / BLOCKSIZE * dims[2] * dims[3]);
  res.emplace_back((dims[0] + 15) / BLOCKSIZE);
  res.emplace_back(BLOCKSIZE);
  res.emplace_back(BLOCKSIZE);

  return res;
}

// NCDHW -> NDHWC
base::IntVector inferShapeOfNDHWC(base::IntVector dims) {
  if (dims.size() < 5) {
    NNDEPLOY_LOGE("dim size (%d) cannot convert to NDHWC.\n", dims.size());
    return base::IntVector();
  }
  base::IntVector res;
  res.resize(5);
  res[0] = dims[0];
  res[1] = dims[2];
  res[2] = dims[3];
  res[3] = dims[4];
  res[4] = dims[1];
  return res;
}

// NCDHW to NCDHW
base::IntVector inferShapeOfNCDHW(base::IntVector dims) {
  if (dims.size() < 5) {
    NNDEPLOY_LOGE("dim size (%d) cannot convert to NCDHW.\n", dims.size());
    return base::IntVector();
  }
  base::IntVector res;
  res.resize(5);
  res[0] = dims[0];
  res[1] = dims[1];
  res[2] = dims[2];
  res[3] = dims[3];
  res[4] = dims[4];
  return res;
}

// NCDHW to NDC1HWC0
base::IntVector inferShapeOfNDC1HWC0(base::IntVector dims) {
  if (dims.size() < 5) {
    NNDEPLOY_LOGE("dim size (%d) cannot convert to NDC1HWC0.\n", dims.size());
    return base::IntVector();
  }
  base::IntVector res;
  res.resize(6);
  res[0] = dims[0];
  res[1] = dims[2];
  res[2] = (dims[1] + BLOCKSIZE - 1) / BLOCKSIZE;
  res[3] = dims[3];
  res[4] = dims[4];
  res[5] = BLOCKSIZE;
  return res;
}

// NCDHW to FZ_3D
base::IntVector inferShapeOfFZ3D(base::IntVector dims) {
  if (dims.size() < 5) {
    NNDEPLOY_LOGE("dim size (%d) cannot convert to FZ_3D.\n", dims.size());
    return base::IntVector();
  }

  int64_t d1 = dims[2];
  int64_t d2 = (dims[1] + BLOCKSIZE - 1) / BLOCKSIZE;
  int64_t d3 = dims[3];
  int64_t d4 = dims[4];
  int64_t d5 = (dims[0] + BLOCKSIZE - 1) / BLOCKSIZE;
  int64_t d6 = BLOCKSIZE;
  int64_t d7 = BLOCKSIZE;

  // The shape of FZ3D is 7D, but the CANN only accept 4D
  // so we should merge 1st, 2nd, 3rd, 4th dimension.
  base::IntVector res;
  res.resize(4);
  res[0] = d1 * d2 * d3 * d4;
  res[1] = d5;
  res[2] = d6;
  res[3] = d7;
  return res;
}

base::IntVector inferShapeofNCHW(base::IntVector dims) {
  return inferShapeLessTo4(dims);
}

base::IntVector inferShapeofND(base::IntVector dims) {
  base::IntVector res;
  res.resize(dims.size());
  for (int j = 0; j < dims.size(); j++) {
    res[j] = dims[j];
  }
  return res;
}

using ShapeInfer = std::function<base::IntVector(base::IntVector dims)>;

struct AclFormatConvert {
  aclFormat dst_format_ = ACL_FORMAT_ND;
  aclFormat src_format_ = ACL_FORMAT_ND;
  ShapeInfer func_ = nullptr;
  char format_name_[128] = {0};
  bool is_padded_ = false;
};

// clang-format off
std::map<aclFormat, std::map<aclFormat, AclFormatConvert>> g_acl_format_convert = {
    // ACL_FORMAT_N、ACL_FORMAT_NC、ACL_FORMAT_NCL、ACL_FORMAT_NCHW、ACL_FORMAT_NHWC、ACL_FORMAT_HWCN
    // -> ACL_FORMAT_ND
    {ACL_FORMAT_ND,
     {{ACL_FORMAT_ND, (AclFormatConvert){ACL_FORMAT_ND, ACL_FORMAT_ND,
                                        inferShapeofND, "ND", false}}}},
    // ACL_FORMAT_N、ACL_FORMAT_NC、ACL_FORMAT_NCL、ACL_FORMAT_NCHW ->
    // ACL_FORMAT_NCHW
    {ACL_FORMAT_NCHW,
      {{ACL_FORMAT_NCHW, (AclFormatConvert){ACL_FORMAT_NCHW, ACL_FORMAT_NCHW,
                                          inferShapeofNCHW, "NCHW", false}}}},
    {ACL_FORMAT_NC1HWC0,
     {{ACL_FORMAT_NCHW, (AclFormatConvert){ACL_FORMAT_NC1HWC0, ACL_FORMAT_NCHW,
                                          inferShape4To5, "NC1HWC0", true}}}},

    {ACL_FORMAT_FRACTAL_NZ,
      {{ACL_FORMAT_ND, (AclFormatConvert){ACL_FORMAT_FRACTAL_NZ, ACL_FORMAT_ND,
                                        inferShapeNDToNZ, "FRACTAL_NZ", true}}}},
    {ACL_FORMAT_FRACTAL_Z,
      {{ACL_FORMAT_NCHW, (AclFormatConvert){ACL_FORMAT_FRACTAL_Z, ACL_FORMAT_NCHW,
                                          inferShapeNDToZ, "FRACTAL_Z", true}}}},
    {ACL_FORMAT_NDHWC,
      {{ACL_FORMAT_NCDHW, (AclFormatConvert){ACL_FORMAT_NDHWC, ACL_FORMAT_NCDHW,
                                           inferShapeOfNDHWC, "NDHWC", false}}}},
    {ACL_FORMAT_NCDHW,
      {{ACL_FORMAT_NCDHW, (AclFormatConvert){ACL_FORMAT_NCDHW, ACL_FORMAT_NCDHW,
                                           inferShapeOfNCDHW, "NCDHW", false}}}},
    {ACL_FORMAT_NDC1HWC0,
      {{ACL_FORMAT_NCDHW,
      (AclFormatConvert){ACL_FORMAT_NDC1HWC0, ACL_FORMAT_NCDHW,
                         inferShapeOfNDC1HWC0, "NDC1HWC0", true}}}},
    {ACL_FRACTAL_Z_3D,
      {{ACL_FORMAT_NCDHW,
      (AclFormatConvert){ACL_FRACTAL_Z_3D, ACL_FORMAT_NCDHW, inferShapeOfFZ3D,
                         "FRACTAL_Z_3D", true}}}},
};
// clang-format on

base::DataType AscendCLOpConvert::convertToDataType(const aclDataType &src) {
  base::DataType dst;
  dst.code_ = base::kDataTypeCodeNotSupport;
  dst.bits_ = 0;
  dst.lanes_ = 0;
  switch (src) {
    case ACL_FLOAT:
      dst = base::dataTypeOf<float>();
      break;
    case ACL_FLOAT16:
      dst.code_ = base::kDataTypeCodeFp;
      dst.bits_ = 16;
      dst.lanes_ = 1;
      break;
    case ACL_INT8:
      dst = base::dataTypeOf<int8_t>();
      break;
    case ACL_INT16:
      dst = base::dataTypeOf<int16_t>();
      break;
    case ACL_INT32:
      dst = base::dataTypeOf<int32_t>();
      break;
    case ACL_INT64:
      dst = base::dataTypeOf<int64_t>();
      break;
    case ACL_UINT8:
      dst = base::dataTypeOf<uint8_t>();
      break;
    case ACL_UINT16:
      dst = base::dataTypeOf<uint16_t>();
      break;
    case ACL_UINT32:
      dst = base::dataTypeOf<uint32_t>();
      break;
    case ACL_UINT64:
      dst = base::dataTypeOf<uint64_t>();
      break;
    case ACL_BOOL:
      dst.code_ = base::kDataTypeCodeInt;
      dst.bits_ = 1;
      dst.lanes_ = 1;
      break;
#if 0
    case ACL_INT4:
      dst.code_ = base::kDataTypeCodeInt;
      dst.bits_ = 4;
      dst.lanes_ = 1;
#endif
    case ACL_UINT1:
      dst.code_ = base::kDataTypeCodeUint;
      dst.bits_ = 1;
      dst.lanes_ = 1;
    case ACL_BF16:
      dst.code_ = base::kDataTypeCodeBFp;
      dst.bits_ = 16;
      dst.lanes_ = 1;
      break;
    default:
      dst.code_ = base::kDataTypeCodeNotSupport;
      dst.bits_ = 0;
      dst.lanes_ = 0;
      break;
  }
  return dst;
}
aclDataType AscendCLOpConvert::convertFromDataType(const base::DataType &src) {
  aclDataType dst = ACL_DT_UNDEFINED;
  if (src.code_ == base::kDataTypeCodeFp) {
    if (src.bits_ == 16) {
      dst = ACL_FLOAT16;
    } else if (src.bits_ == 32) {
      dst = ACL_FLOAT;
    } else if (src.bits_ == 64) {
      dst = ACL_DOUBLE;
    }
  } else if (src.code_ == base::kDataTypeCodeInt) {
    if (src.bits_ == 1) {
      dst = ACL_BOOL;
    }
#if 0
    else if (src.bits_ == 4) {
      dst = ACL_INT4;
    }
#endif
    else if (src.bits_ == 8) {
      dst = ACL_INT8;
    } else if (src.bits_ == 16) {
      dst = ACL_INT16;
    } else if (src.bits_ == 32) {
      dst = ACL_INT32;
    } else if (src.bits_ == 64) {
      dst = ACL_INT64;
    }
  } else if (src.code_ == base::kDataTypeCodeUint) {
    if (src.bits_ == 1) {
      dst = ACL_UINT1;
    } else if (src.bits_ == 8) {
      dst = ACL_UINT8;
    } else if (src.bits_ == 16) {
      dst = ACL_UINT16;
    } else if (src.bits_ == 32) {
      dst = ACL_UINT32;
    } else if (src.bits_ == 64) {
      dst = ACL_UINT64;
    }
  } else if (src.code_ == base::kDataTypeCodeBFp) {
    if (src.bits_ == 16) {
      dst = ACL_BF16;
    }
  }
  return dst;
}

aclFormat AscendCLOpConvert::convertFromDataFormat(base::IntVector shape) {
  if (shape.size() == 3) {
    return ACL_FORMAT_NCL;
  } else if (shape.size() == 4) {
    return ACL_FORMAT_NCHW;
  } else if (shape.size() == 5) {
    return ACL_FORMAT_NCDHW;
  } else {
    return ACL_FORMAT_UNDEFINED;
  }
}
aclFormat AscendCLOpConvert::convertFromDataFormat(
    const base::DataFormat &src) {
  aclFormat dst = ACL_FORMAT_UNDEFINED;
  if (src == base::kDataFormatN) {
    dst = ACL_FORMAT_NC;
  } else if (src == base::kDataFormatNC) {
    dst = ACL_FORMAT_NC;
  } else if (src == base::kDataFormatNCL) {
    dst = ACL_FORMAT_NCL;
  } else if (src == base::kDataFormatNCHW) {
    dst = ACL_FORMAT_NCHW;
  } else if (src == base::kDataFormatOIHW) {
    dst = ACL_FORMAT_NCHW;
  } else if (src == base::kDataFormatNHWC) {
    dst = ACL_FORMAT_NHWC;
  } else if (src == base::kDataFormatNCDHW) {
    dst = ACL_FORMAT_NCDHW;
  } else if (src == base::kDataFormatNDHWC) {
    dst = ACL_FORMAT_NDHWC;
  } else if (src == base::kDataFormatNotSupport) {
    dst = ACL_FORMAT_UNDEFINED;
  } else {
    dst = ACL_FORMAT_ND;
  }
  return dst;
}
base::IntVector AscendCLOpConvert::inferShape(const aclFormat &dst_format,
                                              const aclFormat &src_format,
                                              base::IntVector shape) {
  auto iter = g_acl_format_convert.find(dst_format);
  if (iter != g_acl_format_convert.end()) {
    auto iter_iter = iter->second.find(src_format);
    if (iter_iter == iter->second.end()) {
      iter_iter = iter->second.begin();
    }
    if (iter_iter->second.func_) {
      return iter_iter->second.func_(shape);
    }
  }
  NNDEPLOY_LOGE(
      "unsupport InferShape with dst_format %d src_format %d with shape %s.\n",
      dst_format, src_format, base::vectorToString(shape).c_str());
  return base::IntVector();
}

std::vector<int64_t> AscendCLOpConvert::convertFromShape(base::IntVector &src) {
  std::vector<int64_t> dst;
  for (int i = 0; i < src.size(); i++) {
    dst.push_back((int64_t)src[i]);
  }
  return dst;
}

template <typename T>
aclScalar *AscendCLOpConvert::convertFromScalar(const base::Scalar<T> &src) {
  aclDataType acl_data_type = aclDataTypeOf<T>();
  void *value = (void *)(src.val_);
  aclScalar *acl_scalar = aclCreateScalar(value, acl_data_type);
  return acl_scalar;
}
template <typename T>
aclScalar *AscendCLOpConvert::convertFromScalar(float src) {
  aclDataType acl_data_type = aclDataTypeOf<T>();
  void *value = nullptr;
  aclScalar *acl_scalar = nullptr;
  switch (acl_data_type) {
    case ACL_FLOAT:
      value = malloc(sizeof(float));
      *(float *)value = src;
      break;
    case ACL_FLOAT16:
      value = malloc(sizeof(float) >> 1);
      base::convertFromFloatToFp16(&src, value, 1);
      break;
    case ACL_INT8:
      value = malloc(sizeof(int8_t));
      *(int8_t *)value = base::saturate_cast<int8_t>(src);
      break;
    case ACL_INT16:
      value = malloc(sizeof(int16_t));
      *(int16_t *)value = base::saturate_cast<int16_t>(src);
      break;
    case ACL_INT32:
      value = malloc(sizeof(int32_t));
      *(int32_t *)value = base::saturate_cast<int32_t>(src);
      break;
    case ACL_INT64:
      value = malloc(sizeof(int64_t));
      *(int64_t *)value = base::saturate_cast<int64_t>(src);
      break;
    case ACL_UINT8:
      value = malloc(sizeof(uint8_t));
      *(uint8_t *)value = base::saturate_cast<uint8_t>(src);
      break;
    case ACL_UINT16:
      value = malloc(sizeof(uint16_t));
      *(uint16_t *)value = base::saturate_cast<uint16_t>(src);
      break;
    case ACL_UINT32:
      value = malloc(sizeof(uint32_t));
      *(uint32_t *)value = base::saturate_cast<uint32_t>(src);
      break;
    case ACL_UINT64:
      value = malloc(sizeof(uint64_t));
      *(uint64_t *)value = base::saturate_cast<uint64_t>(src);
      break;
    case ACL_BOOL:
      value = malloc(sizeof(int8_t));
      *(int8_t *)value = base::saturate_cast<int8_t>(src);
      break;
#if 0
    case ACL_INT4:
      value = malloc(sizeof(int8_t));
      *(int8_t *)value = base::saturate_cast<int8_t>(src);
      break;
#endif
    case ACL_UINT1:
      value = malloc(sizeof(uint8_t));
      *(uint8_t *)value = base::saturate_cast<uint8_t>(src);
      break;
    case ACL_BF16:
      value = malloc(sizeof(float) >> 1);
      base::convertFromFloatToBfp16(&src, value, 1);
      break;
    default:
      break;
  }
  if (value != nullptr) {
    acl_scalar = aclCreateScalar(value, acl_data_type);
  }
  if (value != nullptr) {
    free(value);
  }
  return acl_scalar;
}
aclScalar *AscendCLOpConvert::convertFromScalar(
    float src, const base::DataType &data_type) {
  aclDataType acl_data_type = convertFromDataType(data_type);
  void *value = nullptr;
  aclScalar *acl_scalar = nullptr;
  switch (acl_data_type) {
    case ACL_FLOAT:
      value = malloc(sizeof(float));
      *(float *)value = src;
      break;
    case ACL_FLOAT16:
      value = malloc(sizeof(float) >> 1);
      base::convertFromFloatToFp16(&src, value, 1);
      break;
    case ACL_INT8:
      value = malloc(sizeof(int8_t));
      *(int8_t *)value = base::saturate_cast<int8_t>(src);
      break;
    case ACL_INT16:
      value = malloc(sizeof(int16_t));
      *(int16_t *)value = base::saturate_cast<int16_t>(src);
      break;
    case ACL_INT32:
      value = malloc(sizeof(int32_t));
      *(int32_t *)value = base::saturate_cast<int32_t>(src);
      break;
    case ACL_INT64:
      value = malloc(sizeof(int64_t));
      *(int64_t *)value = base::saturate_cast<int64_t>(src);
      break;
    case ACL_UINT8:
      value = malloc(sizeof(uint8_t));
      *(uint8_t *)value = base::saturate_cast<uint8_t>(src);
      break;
    case ACL_UINT16:
      value = malloc(sizeof(uint16_t));
      *(uint16_t *)value = base::saturate_cast<uint16_t>(src);
      break;
    case ACL_UINT32:
      value = malloc(sizeof(uint32_t));
      *(uint32_t *)value = base::saturate_cast<uint32_t>(src);
      break;
    case ACL_UINT64:
      value = malloc(sizeof(uint64_t));
      *(uint64_t *)value = base::saturate_cast<uint64_t>(src);
      break;
    case ACL_BOOL:
      value = malloc(sizeof(int8_t));
      *(int8_t *)value = base::saturate_cast<int8_t>(src);
      break;
#if 0
    case ACL_INT4:
      value = malloc(sizeof(int8_t));
      *(int8_t *)value = base::saturate_cast<int8_t>(src);
      break;
#endif
    case ACL_UINT1:
      value = malloc(sizeof(uint8_t));
      *(uint8_t *)value = base::saturate_cast<uint8_t>(src);
      break;
    case ACL_BF16:
      value = malloc(sizeof(float) >> 1);
      base::convertFromFloatToBfp16(&src, value, 1);
      break;
    default:
      break;
  }
  if (value != nullptr) {
    acl_scalar = aclCreateScalar(value, acl_data_type);
  }
  if (value != nullptr) {
    free(value);
  }
  return acl_scalar;
}
// template <typename T>
// aclScalarList *AscendCLOpConvert::convertFromScalar(
//     const std::vector<base::Scalar<T>> &src);

aclIntArray *AscendCLOpConvert::convertFromIntVector(
    const std::vector<int> &src) {
  std::vector<int64_t> tmp(src.begin(), src.end());
  aclIntArray *dst = aclCreateIntArray(tmp.data(), tmp.size());
  return dst;
}
aclFloatArray *AscendCLOpConvert::convertFromFloatVector(
    const std::vector<float> &src) {
  aclFloatArray *dst = aclCreateFloatArray(src.data(), src.size());
  return dst;
}
// aclBoolArray *AscendCLOpConvert::convertFromBoolVector(
//     const std::vector<bool> &src) {
//   const bool *value = src.data();
//   aclBoolArray *dst = aclCreateBoolArray(value, src.size());
//   return dst;
// }
// aclFp16Array *AscendCLOpConvert::convertFromFp16Vector(
//     const std::vector<half_float::half> &src) {
//   aclFp16Array *dst = aclCreateFp16Array(src.data(), src.size());
//   return dst;
// }
// aclBf16Array *AscendCLOpConvert::convertFromBfp16Vector(
//     const std::vector<base::bfp16_t> &src) {
//   aclBf16Array *dst = aclCreateBfp16Array(src.data(), src.size());
//   return dst;
// }

aclTensor *AscendCLOpConvert::convertFromTensor(const device::Tensor *src,
                                                aclFormat dst_data_format) {
  base::DeviceType device_type = src->getDeviceType();
  if (device_type.code_ != base::kDeviceTypeCodeAscendCL) {
    NNDEPLOY_LOGE("device type is not Ascend when convertFromTensor.\n");
    return nullptr;
  }

  base::DataType src_data_type = src->getDataType();
  aclDataType dst_data_type =
      AscendCLOpConvert::convertFromDataType(src_data_type);

  base::DataFormat src_data_format = src->getDataFormat();
  aclFormat acl_data_format =
      AscendCLOpConvert::convertFromDataFormat(src_data_format);
  base::IntVector src_shape = src->getShape();
  base::IntVector dim = AscendCLOpConvert::inferShape(
      dst_data_format, acl_data_format, src_shape);
  std::vector<int64_t> dst_dim = AscendCLOpConvert::convertFromShape(dim);

  base::SizeVector src_stride = src->getStride();
  std::vector<int64_t> dst_stride;
  if (src_stride.empty()) {
    dst_stride.resize(dst_dim.size(), 1);
    for (int64_t i = dst_dim.size() - 2; i >= 0; i--) {
      dst_stride[i] = dst_dim[i + 1] * dst_stride[i + 1];
    }
  } else {
    for (auto iter : src_stride) {
      dst_stride.push_back((int64_t)iter);
    }
  }

  int64_t offset = 0;
  void *data = src->getData();

  aclTensor *dst = aclCreateTensor(
      dst_dim.data(), dst_dim.size(), dst_data_type, dst_stride.data(), offset,
      dst_data_format, dst_dim.data(), dst_dim.size(), data);
  if (dst == nullptr) {
    NNDEPLOY_LOGE("aclCreateTensor failed when convertFromTensor.\n");
  }

#if 0
  std::string name_ = src->getName();
  if (name_ == "/model.2/cv1/act/Mul_output_0" ||
      name_ == "/model.2/Split_output_0" ||
      name_ == "/model.2/Split_output_1") {
    NNDEPLOY_LOGI("%s\n", base::dataFormatToString(src_data_format).c_str());
    NNDEPLOY_LOGI("src_shape: %d, %d, %d, %d\n", src_shape[0], src_shape[1],
                  src_shape[2], src_shape[3]);
    NNDEPLOY_LOGI("dim: %d, %d, %d, %d\n", dim[0], dim[1], dim[2], dim[3]);
    NNDEPLOY_LOGI("dst_stride: %d, %d, %d, %d\n", dst_stride[0],
    dst_stride[1],
                  dst_stride[2], dst_stride[3]);
    NNDEPLOY_LOGI("data: %p\n", data);
    src->getDesc().print();
  }
#endif

  return dst;
}
aclTensorList *AscendCLOpConvert::AscendCLOpConvert::convertFromTensor(
    const std::vector<device::Tensor *> &src, aclFormat dst_data_format) {
  static std::vector<aclTensor *> tensor_list;
  tensor_list.clear();
  for (size_t i = 0; i < src.size(); i++) {
    aclTensor *tensor = convertFromTensor(src[i], dst_data_format);
    tensor_list.push_back(tensor);
  }
  aclTensorList *acl_tensor_list =
      aclCreateTensorList(tensor_list.data(), tensor_list.size());
  if (acl_tensor_list == nullptr) {
    NNDEPLOY_LOGE("aclCreateTensorList failed when convertFromTensor.\n");
  }
  return acl_tensor_list;
}

}  // namespace op
}  // namespace nndeploy
