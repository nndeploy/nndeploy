
#include "nndeploy/inference/openvino/openvino_convert.h"

#include "nndeploy/inference/openvino/openvino_inference.h"

namespace nndeploy {
namespace inference {

base::DataType OpenVinoConvert::convertToDataType(
    const nvinfer1::DataType &src) {
  base::DataType dst;
  switch (src) {
    case nvinfer1::DataType::kFLOAT:
      dst.code_ = base::kDataTypeCodeFp;
      dst.bits_ = 32;
      dst.lanes_ = 1;
    case nvinfer1::DataType::kHALF:
      dst.code_ = base::kDataTypeCodeBFp;
      dst.bits_ = 16;
      dst.lanes_ = 1;
    case nvinfer1::DataType::kINT32:
      dst.code_ = base::kDataTypeCodeInt;
      dst.bits_ = 32;
      dst.lanes_ = 1;
    case nvinfer1::DataType::kINT8:
      dst.code_ = base::kDataTypeCodeInt;
      dst.bits_ = 8;
      dst.lanes_ = 1;
    case nvinfer1::DataType::kUINT8:
      dst.code_ = base::kDataTypeCodeUint;
      dst.bits_ = 8;
      dst.lanes_ = 1;
    case nvinfer1::DataType::kBOOL:
      dst.code_ = base::kDataTypeCodeUint;
      dst.bits_ = 8;
      dst.lanes_ = 1;
    default:
      dst.code_ = base::kDataTypeCodeFp;
      dst.bits_ = 32;
      dst.lanes_ = 1;
  }
  return dst;
}

nvinfer1::DataType OpenVinoConvert::convertFromDataType(base::DataType &src) {
  nvinfer1::DataType dst;
  if (src.code_ == base::kDataTypeCodeFp && src.bits_ == 32 &&
      src.lanes_ == 1) {
    dst = nvinfer1::DataType::kFLOAT;
  } else if (src.code_ == base::kDataTypeCodeBFp && src.bits_ == 16 &&
             src.lanes_ == 1) {
    dst = nvinfer1::DataType::kHALF;
  } else if (src.code_ == base::kDataTypeCodeInt && src.bits_ == 32 &&
             src.lanes_ == 1) {
    dst = nvinfer1::DataType::kINT32;
  } else if (src.code_ == base::kDataTypeCodeInt && src.bits_ == 8 &&
             src.lanes_ == 1) {
    dst = nvinfer1::DataType::kINT8;
  } else if (src.code_ == base::kDataTypeCodeUint && src.bits_ == 8 &&
             src.lanes_ == 1) {
    dst = nvinfer1::DataType::kUINT8;
  } else {
    dst = nvinfer1::DataType::kFLOAT;
  }
  return dst;
}

base::DataFormat OpenVinoConvert::convertToDataFormat(
    const nvinfer1::TensorFormat &src) {
  base::DataFormat dst;
  switch (src) {
    case nvinfer1::TensorFormat::kLINEAR:
      dst = base::kDataFormatNCHW;
      break;
    case nvinfer1::TensorFormat::kCHW4:
      dst = base::kDataFormatNC4HW;
      break;
    default:
      dst = base::kDataFormatNCHW;
      break;
  }
  return dst;
}

base::IntVector OpenVinoConvert::convertToShape(const nvinfer1::Dims &src) {
  base::IntVector dst;
  int src_size = src.nbDims;
  for (int i = 0; i < src_size; ++i) {
    dst.push_back(src.d[i]);
  }
  return dst;
}

nvinfer1::Dims OpenVinoConvert::convertFromShape(const base::IntVector &src) {
  int src_size = src.size();
  nvinfer1::Dims dst;
  dst.nbDims = src_size;
  for (int i = 0; i < src_size; ++i) {
    dst.d[i] = src[i];
  }
  return dst;
}

base::Status OpenVinoConvert::convertFromInferenceParam() {
  if (ov_device_type.find("HETERO") != std::string::npos) {
    auto supported_ops = core_.query_model(model, ov_device_type);
    for (auto &&op : model->get_ops()) {
      auto &affinity = supported_ops[op->get_friendly_name()];
      op->get_rt_info()["affinity"] = affinity;
    }
  }

  if (openvino_inference_param->hint_ == "UNDEFINED") {
    if (ov_device_type == "CPU") {
      properties["INFERENCE_NUM_THREADS"] =
          openvino_inference_param->num_thread_;
    }
    if (openvino_inference_param->num_streams_ == -1) {
      properties["NUM_STREAMS"] = ov::streams::AUTO;
    } else if (openvino_inference_param->num_streams_ == -2) {
      properties["NUM_STREAMS"] = ov::streams::NUMA;
    } else if (openvino_inference_param->num_streams_ > 0) {
      properties["NUM_STREAMS"] = openvino_inference_param->num_streams_;
    }

    if (openvino_inference_param->affinity_ == "YES") {
      properties["AFFINITY"] = "CORE";
    } else if (openvino_inference_param->affinity_ == "NO") {
      properties["AFFINITY"] = "NONE";
    } else if (openvino_inference_param->affinity_ == "NUMA") {
      properties["AFFINITY"] = "NUMA";
    } else if (openvino_inference_param->affinity_ == "HYBRID_AWARE") {
      properties["AFFINITY"] = "HYBRID_AWARE";
    }
  } else if (openvino_inference_param->hint_ == "LATENCY") {
    properties.emplace(
        ov::hint::performance_mode(ov::hint::PerformanceMode::LATENCY));
  } else if (openvino_inference_param->hint_ == "THROUGHPUT") {
    properties.emplace(
        ov::hint::performance_mode(ov::hint::PerformanceMode::THROUGHPUT));
  } else if (openvino_inference_param->hint_ == "CUMULATIVE_THROUGHPUT") {
    properties.emplace(ov::hint::performance_mode(
        ov::hint::PerformanceMode::CUMULATIVE_THROUGHPUT));
  }
}

}  // namespace inference
}  // namespace nndeploy
