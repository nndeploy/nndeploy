
#include "nndeploy/op/x86/op_convert.h"

namespace nndeploy {

namespace op {

base::DataType X86OpConvert::convertToDataType(
    const dnnl::memory::data_type &src) {
  base::DataType dst;
  switch (src) {
    case dnnl::memory::data_type::f32:
      dst.code_ = base::DataTypeCode::kDataTypeCodeFp;
      dst.bits_ = 32;
      dst.lanes_ = 1;
      break;
    case dnnl::memory::data_type::f16:
      dst.code_ = base::DataTypeCode::kDataTypeCodeFp;
      dst.bits_ = 16;
      dst.lanes_ = 1;
      break;
    case dnnl::memory::data_type::bf16:
      dst.code_ = base::DataTypeCode::kDataTypeCodeBFp;
      dst.bits_ = 16;
      dst.lanes_ = 1;
      break;
    case dnnl::memory::data_type::s8:
      dst.code_ = base::DataTypeCode::kDataTypeCodeInt;
      dst.bits_ = 8;
      dst.lanes_ = 1;
      break;
    case dnnl::memory::data_type::u8:
      dst.code_ = base::DataTypeCode::kDataTypeCodeUint;
      dst.bits_ = 8;
      dst.lanes_ = 1;
      break;
    case dnnl::memory::data_type::s32:
      dst.code_ = base::kDataTypeCodeInt;
      dst.bits_ = 32;
      dst.lanes_ = 1;
      break;
    case dnnl::memory::data_type::f64:
      dst.code_ = base::kDataTypeCodeFp;
      dst.bits_ = 64;
      dst.lanes_ = 1;
      break;
    case dnnl::memory::data_type::s4:
      dst.code_ = base::kDataTypeCodeInt;
      dst.bits_ = 4;
      dst.lanes_ = 1;
      break;
    case dnnl::memory::data_type::u4:
      dst.code_ = base::kDataTypeCodeUint;
      dst.bits_ = 4;
      dst.lanes_ = 1;
      break;

    case dnnl::memory::data_type::undef:
      dst.code_ = base::kDataTypeCodeNotSupport;
      dst.bits_ = 0;
      dst.lanes_ = 0;
      break;
    default:
      NNDEPLOY_LOGE("Unsupported dnnl::datatype: %d\n", src);
  }
  return dst;
}

dnnl::memory::data_type X86OpConvert::convertFromDataType(
    const base::DataType &src) {
  dnnl::memory::data_type dst = dnnl::memory::data_type::undef;
  if (src.code_ == base::kDataTypeCodeFp) {
    if (src.bits_ == 16) {
      dst = dnnl::memory::data_type::f16;
    } else if (src.bits_ == 32) {
      dst = dnnl::memory::data_type::f32;
    } else if (src.bits_ == 64) {
      dst = dnnl::memory::data_type::f64;
    }
  } else if (src.code_ == base::kDataTypeCodeInt) {
    if (src.bits_ == 8) {
      dst = dnnl::memory::data_type::s8;
    } else if (src.bits_ == 32) {
      dst = dnnl::memory::data_type::s32;
    } else if (src.bits_ == 4) {
      dst = dnnl::memory::data_type::s4;
    }
  } else if (src.code_ == base::kDataTypeCodeUint) {
    if (src.bits_ == 8) {
      dst = dnnl::memory::data_type::u8;
    } else if (src.bits_ == 4) {
      dst = dnnl::memory::data_type::u4;
    }
  } else if (src.code_ == base::kDataTypeCodeBFp) {
    if (src.bits_ == 16) {
      dst = dnnl::memory::data_type::bf16;
    }
  }
  return dst;
}

dnnl::memory::dims X86OpConvert::convertFromShape(const base::IntVector &src) {
  dnnl::memory::dims dnnl_dims(src.begin(), src.end());
  return dnnl_dims;
}
dnnl::memory::format_tag X86OpConvert::convertFromDataFormat(
    const base::DataFormat &src) {
  dnnl::memory::format_tag dst = dnnl::memory::format_tag::undef;
  switch (src) {
    case base::kDataFormatN:
      dst = dnnl::memory::format_tag::x;
      break;
    case base::kDataFormatNC:
      dst = dnnl::memory::format_tag::nc;
      break;
    case base::kDataFormatNCHW:
      dst = dnnl::memory::format_tag::nchw;
      break;
    case base::kDataFormatOIHW:
      dst = dnnl::memory::format_tag::oihw;
      break;
    case base::kDataFormatNHWC:
      dst = dnnl::memory::format_tag::nhwc;
      break;
    case base::kDataFormatNCDHW:
      dst = dnnl::memory::format_tag::ncdhw;
      break;
    case base::kDataFormatNDHWC:
      dst = dnnl::memory::format_tag::ndhwc;
      break;
    case base::kDataFormatAuto:
    case base::kDataFormatNotSupport:
    default:
      NNDEPLOY_LOGE("Unsupported nndeploy::base::dataformat: %d\n", src);
      break;
  }
  return dst;
}

base::DataFormat X86OpConvert::convertToDataFormat(
    const dnnl::memory::format_tag &src) {
  base::DataFormat dst = base::kDataFormatNotSupport;
  switch (src) {
    case dnnl::memory::format_tag::x:
      dst = base::kDataFormatN;
      break;
    case dnnl::memory::format_tag::nc:
      dst = base::kDataFormatNC;
      break;
    case dnnl::memory::format_tag::nchw:
      dst = base::kDataFormatNCHW;
      break;
    // dnnl对于nchw和oihw使用相同枚举值
    // case dnnl::memory::format_tag::oihw:
    //   dst = base::kDataFormatOIHW;
    //   break;
    case dnnl::memory::format_tag::nhwc:
      dst = base::kDataFormatNHWC;
      break;
    case dnnl::memory::format_tag::ncdhw:
      dst = base::kDataFormatNCDHW;
      break;
    case dnnl::memory::format_tag::ndhwc:
      dst = base::kDataFormatNDHWC;
      break;
    case dnnl::memory::format_tag::undef:
    case dnnl::memory::format_tag::any:
    default:
      NNDEPLOY_LOGE("Unsupported dnnl::memory::format_tag: %d\n", src);
      break;
  }
  return dst;
}

}  // namespace op
}  // namespace nndeploy