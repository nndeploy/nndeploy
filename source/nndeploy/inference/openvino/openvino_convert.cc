
#include "nndeploy/inference/openvino/openvino_convert.h"

#include "nndeploy/inference/openvino/openvino_inference.h"

namespace nndeploy {
namespace inference {

std::string OpenVinoConvert::OpenVinoConvert::convertFromDeviceType(
    const base::device &src, const base::device &srcs) {
  std::string dst = "CPU";
  if (srcs.empty()) {
    if (src.code_ == kDeviceTypeCodeOpenCL) {
      dst = "GPU";
    }
  } else {
    if (device::isHostDeviceType(src) &&
        srcs[0].code_ == kDeviceTypeCodeOpenCL) {
      dst = "CPU:GPU";
    }
    if (device::isHostDeviceType(srcs[0]) &&
        src.code_ == kDeviceTypeCodeOpenCL) {
      dst = "GPU:CPU";
    }
  }
  return dst;
}

base::DataType OpenVinoConvert::convertToDataType(
    const ov::element::Type &src) {
  base::DataType dst;
  switch (src) {
    case ov::element::bf16:
      dst.code_ = base::kDataTypeCodeBFp;
      dst.bits_ = 16;
      dst.lanes_ = 1;
    case ov::element::f16:
      dst.code_ = base::kDataTypeCodeFp;
      dst.bits_ = 16;
      dst.lanes_ = 1;
    case ov::element::f32:
      dst.code_ = base::kDataTypeCodeFp;
      dst.bits_ = 32;
      dst.lanes_ = 1;
    case ov::element::f64:
      dst.code_ = base::kDataTypeCodeFp;
      dst.bits_ = 64;
      dst.lanes_ = 1;
    case ov::element::u4:
      dst.code_ = base::kDataTypeCodeUint;
      dst.bits_ = 4;
      dst.lanes_ = 1;
    case ov::element::u8:
      dst.code_ = base::kDataTypeCodeUint;
      dst.bits_ = 8;
      dst.lanes_ = 1;
    case ov::element::u16:
      dst.code_ = base::kDataTypeCodeUint;
      dst.bits_ = 16;
      dst.lanes_ = 1;
    case ov::element::u32:
      dst.code_ = base::kDataTypeCodeUint;
      dst.bits_ = 32;
      dst.lanes_ = 1;
    case ov::element::u64:
      dst.code_ = base::kDataTypeCodeUint;
      dst.bits_ = 64;
      dst.lanes_ = 1;
    case ov::element::i4:
      dst.code_ = base::kDataTypeCodeInt;
      dst.bits_ = 8;
      dst.lanes_ = 1;
    case ov::element::i8:
      dst.code_ = base::kDataTypeCodeInt;
      dst.bits_ = 8;
      dst.lanes_ = 1;
    case ov::element::i16:
      dst.code_ = base::kDataTypeCodeInt;
      dst.bits_ = 16;
      dst.lanes_ = 1;
    case ov::element::i32:
      dst.code_ = base::kDataTypeCodeInt;
      dst.bits_ = 32;
      dst.lanes_ = 1;
    case ov::element::i64:
      dst.code_ = base::kDataTypeCodeInt;
      dst.bits_ = 64;
      dst.lanes_ = 1;
    default:
      dst.code_ = base::kDataTypeCodeOpaqueHandle;
      dst.bits_ = src.bitwidth();
      dst.lanes_ = 1;
  }
  return dst;
}

ov::element::Type OpenVinoConvert::convertFromDataType(base::DataType &src) {
  ONNXTensorElementDataType dst;
  if (src.code_ == base::kDataTypeCodeFp && src.lanes_ == 1) {
    if (src.bits_ == 16) {
      dst = ov::element::f16;
    } else if (src.bits_ == 32) {
      dst = ov::element::f32;
    } else if (src.bits_ == 64) {
      dst = ov::element::f64;
    } else {
      dst = ov::element::undefined;
    }
  } else if (src.code_ == base::kDataTypeCodeBFp && src.bits_ == 16 &&
             src.lanes_ == 1) {
    dst = ov::element::bf16;
  } else if (src.code_ == base::kDataTypeCodeInt && src.lanes_ == 1) {
    if (src.bits_ == 8) {
      dst = ov::element::i8;
    } else if (src.bits_ == 16) {
      dst = ov::element::i16;
    } else if (src.bits_ == 32) {
      dst = ov::element::i32;
    } else if (src.bits_ == 64) {
      dst = ov::element::i64;
    } else {
      dst = ov::element::undefined;
    }
  } else if (src.code_ == base::kDataTypeCodeUint && src.lanes_ == 1) {
    if (src.bits_ == 8) {
      dst = ov::element::u8;
    } else if (src.bits_ == 16) {
      dst = ov::element::u16;
    } else if (src.bits_ == 32) {
      dst = ov::element::u32;
    } else if (src.bits_ == 64) {
      dst = ov::element::u64;
    } else {
      dst = ov::element::undefined;
    }
  } else {
    dst = ov::element::undefined;
  }
  return dst;
}

base::DataFormat OpenVinoConvert::getDataFormatByShape(
    const base::IntVector &src) {
  base::DataFormat dst = base::kDataFormatNotSupport;
  if (src.size() == 5) {
    dst = base::kDataFormatNCDHW;
  } else if (src.size() == 4) {
    dst = base::kDataFormatNCHW;
  } else if (src.size() == 3) {
    dst = base::kDataFormatNCHW;
  } else if (src.size() == 2) {
    dst = base::kDataFormatNC;
  } else if (src.size() == 1) {
    dst = base::kDataFormatScalar;
  } else {
    dst = base::kDataFormatNotSupport;
  }
  return dst;
}

base::IntVector OpenVinoConvert::convertToShape(const ov::PartialShape &src) {
  base::IntVector dst;
  for (int i = 0; i < src.size(); ++i) {
    auto dim = src[i];
    if (dim.is_dynamic()) {
      dst.push_back(-1);  // -1 means dynamic shape
    } else {
      dst.push_back(dim.get_length());
    }
  }
  return dst;
}

ov::PartialShape OpenVinoConvert::convertFromShape(const base::IntVector &src) {
  int src_size = src.size();
  std::vector<ov::Dimension> dims;
  for (size_t i = 0; i < src_size; ++i) {
    dims.emplace_back(ov::Dimension(src[i]));
  }
  return ov::PartialShape(dims);
}

base::SizeVector OpenVinoConvert::convertToStride(const ov::Strides &src) {
  base::SizeVector dst;
  for (int i = 0; i < src.size(); ++i) {
    auto dim = src[i];
    dst.push_back(dim);
  }
  return dst;
}

ov::Strides OpenVinoConvert::convertFromShape(const base::SizeVector &src) {
  return ov::Strides(src);
}

base::Status OpenVinoConvert::convertFromInferenceParam(
    OpenVinoInferenceParam *src, std::string &dst_device_type,
    ov::AnyMap &dst_properties) {
  dst_device_type = OpenVinoConvert::convertFromDeviceType(src->device_type_,
                                                           src->device_types_);
  if (dst_device_type.find("HETERO") != std::string::npos) {
    auto supported_ops = core_.query_model(model, dst_device_type);
    for (auto &&op : model->get_ops()) {
      auto &affinity = supported_ops[op->get_friendly_name()];
      op->get_rt_info()["affinity"] = affinity;
    }
  }

  if (openvino_inference_param->hint_ == "UNDEFINED") {
    if (dst_device_type == "CPU") {
      dst_properties["INFERENCE_NUM_THREADS"] =
          openvino_inference_param->num_thread_;
    }
    if (openvino_inference_param->num_streams_ == -1) {
      dst_properties["NUM_STREAMS"] = ov::streams::AUTO;
    } else if (openvino_inference_param->num_streams_ == -2) {
      dst_properties["NUM_STREAMS"] = ov::streams::NUMA;
    } else if (openvino_inference_param->num_streams_ > 0) {
      dst_properties["NUM_STREAMS"] = openvino_inference_param->num_streams_;
    }

    if (openvino_inference_param->affinity_ == "YES") {
      dst_properties["AFFINITY"] = "CORE";
    } else if (openvino_inference_param->affinity_ == "NO") {
      dst_properties["AFFINITY"] = "NONE";
    } else if (openvino_inference_param->affinity_ == "NUMA") {
      dst_properties["AFFINITY"] = "NUMA";
    } else if (openvino_inference_param->affinity_ == "HYBRID_AWARE") {
      dst_properties["AFFINITY"] = "HYBRID_AWARE";
    }
  } else if (openvino_inference_param->hint_ == "LATENCY") {
    dst_properties.emplace(
        ov::hint::performance_mode(ov::hint::PerformanceMode::LATENCY));
  } else if (openvino_inference_param->hint_ == "THROUGHPUT") {
    dst_properties.emplace(
        ov::hint::performance_mode(ov::hint::PerformanceMode::THROUGHPUT));
  } else if (openvino_inference_param->hint_ == "CUMULATIVE_THROUGHPUT") {
    dst_properties.emplace(ov::hint::performance_mode(
        ov::hint::PerformanceMode::CUMULATIVE_THROUGHPUT));
  }

  return status;
}

ov::Tensor OpenVinoConvert::convertFromTensor(device::Tensor *src) {
  if (src == nullptr) {
    return ov::Tensor();
  }
  auto data_type = OpenVinoConvert::convertFromDataType(src->getDataType());
  auto shape = OpenVinoConvert::convertFromShape(src->getShape());
  void *data = src->getPtr();
  ov::Tensor ov_tensor(data_type, shape, data);
  return ov_tensor;
}

device::Tensor *OpenVinoConvert::convertToTensor(ov::Tensor &src,
                                                 const std::string &name) {
  device::Device *device = device::getDefaultHostDevice();
  TensorDesc desc;
  desc.data_type_ = OpenVinoConvert::convertToDataType(src.get_element_type());
  des.shape_ = OpenVinoConvert::convertToShape(src.get_shape());
  desc.format_ = OpenVinoConvert::getDataFormatByShape(des.shape);
  desc.stride_ = OpenVinoConvert::convertToStride(des.get_strides());
  void *data = src.data();
  device::Tensor *tensor = new device::Tensor(device, desc, data, name);
  return tensor;
}

}  // namespace inference
}  // namespace nndeploy
