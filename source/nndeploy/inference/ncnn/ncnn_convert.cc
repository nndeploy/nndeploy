

#include "nndeploy/inference/ncnn/ncnn_convert.h"

namespace nndeploy {
namespace inference {

base::DataType NcnnConvert::convertToDataType(const int &src) {
  base::DataType dst;
  if (src == 4) {
    dst.code_ = base::kDataTypeCodeFp;  // Int
    dst.bits_ = src * 8;
    dst.lanes_ = 1;
  } else if (src == 2) {
    dst.code_ = base::kDataTypeCodeFp;  // BFp
    dst.bits_ = src * 8;
    dst.lanes_ = 1;
  } else if (src == 1) {
    dst.code_ = base::kDataTypeCodeInt;  // Uint
    dst.bits_ = src * 8;
    dst.lanes_ = 1;
  } else {
    dst = base::dataTypeOf<float>();
    NNDEPLOY_LOGE("Not support data type[%d]!\n", src);
  }

  return dst;
}

base::DataFormat NcnnConvert::convertToDataFormat(const int &elempack,
                                                  const int &dims, const int &w,
                                                  const int &h, const int &d,
                                                  const int &c,
                                                  const size_t &cstep) {
  base::DataFormat dst;
  if (dims == 4) {
    dst = base::kDataFormatNCDHW;
  } else if (dims == 3) {
    dst = base::kDataFormatNCHW;
  } else if (dims == 2) {
    dst = base::kDataFormatNHW;
  } else if (dims == 1) {
    dst = base::kDataFormatNC;
  } else {
    dst = base::kDataFormatNotSupport;
    NNDEPLOY_LOGE("Not support data format[%d]!\n", dims);
  }
  return dst;
}

base::IntVector NcnnConvert::convertToShape(const int &dims, const int &w,
                                            const int &h, const int &d,
                                            const int &c, const size_t &cstep) {
  base::IntVector dst;
  dst.emplace_back(1);
  if (dims == 4) {
    dst.emplace_back(c);
    dst.emplace_back(d);
    dst.emplace_back(h);
    dst.emplace_back(w);
  } else if (dims == 3) {
    dst.emplace_back(c);
    dst.emplace_back(h);
    dst.emplace_back(w);
  } else if (dims == 2) {
    dst.emplace_back(h);
    dst.emplace_back(w);
  } else if (dims == 1) {
    dst.emplace_back(w);
  } else {
    dst.clear();
    NNDEPLOY_LOGE("Not support data format[%d]!\n", dims);
  }
  return dst;
}

base::Status NcnnConvert::convertFromInferenceParam(
    inference::NcnnInferenceParam *src, ncnn::Option &dst) {
  base::Status status = base::kStatusCodeOk;
  dst.lightmode = src->lightmode_;

  dst.num_threads = src->num_thread_;

  dst.openmp_blocktime = src->openmp_blocktime_;

  dst.use_winograd_convolution = src->use_winograd_convolution_;

  dst.use_sgemm_convolution = src->use_sgemm_convolution_;

  dst.use_int8_inference = src->use_int8_inference_;

  dst.use_vulkan_compute =
      src->device_type_.code_ == base::kDeviceTypeCodeVulkan;

  dst.use_bf16_storage = src->use_bf16_storage_;

  dst.use_fp16_packed = src->use_fp16_packed_;
  dst.use_fp16_storage = src->use_fp16_storage_;
  dst.use_fp16_arithmetic = src->use_fp16_arithmetic_;
  dst.use_int8_packed = src->use_int8_packed_;
  dst.use_int8_storage = src->use_int8_storage_;
  dst.use_int8_arithmetic = src->use_int8_arithmetic_;

  dst.use_packing_layout = src->use_packing_layout_;

  dst.use_shader_pack8 = src->use_shader_pack8_;

  dst.use_subgroup_basic = src->use_subgroup_basic_;
  dst.use_subgroup_vote = src->use_subgroup_vote_;
  dst.use_subgroup_ballot = src->use_subgroup_ballot_;
  dst.use_subgroup_shuffle = src->use_subgroup_shuffle_;

  dst.use_image_storage = src->use_image_storage_;
  dst.use_tensor_storage = src->use_tensor_storage_;

  dst.use_reserved_0 = src->use_reserved_0_;

  dst.flush_denormals = src->flush_denormals_;

  dst.use_local_pool_allocator = src->use_local_pool_allocator_;

  // enable local memory optimization for gpu inference
  dst.use_shader_local_memory = src->use_local_pool_allocator_;

  // enable cooperative matrix optimization for gpu inference
  dst.use_cooperative_matrix = src->use_cooperative_matrix_;

  // more fine-grained control of winograd convolution
  dst.use_winograd23_convolution = src->use_winograd23_convolution_;
  dst.use_winograd43_convolution = src->use_winograd43_convolution_;
  dst.use_winograd63_convolution = src->use_winograd63_convolution_;

  // this option is turned on for A53/A55 automatically
  // but you can force this on/off if you wish
  dst.use_a53_a55_optimized_kernel = src->use_a53_a55_optimized_kernel_;

  return base::kStatusCodeOk;
}

base::Status NcnnConvert::matConvertToTensor(ncnn::Mat &src,
                                             const std::string &name,
                                             device::Tensor *dst) {
  dst->destory();
  device::Device *device = device::getDefaultHostDevice();
  base::DataType data_type = NcnnConvert::convertToDataType(src.elemsize);
  base::DataFormat data_format =
      convertToDataFormat(src.elempack, src.dims, src.w, src.h, src.d, src.c,
                          src.cstep);  // 目前只使用了shape.dims
  base::IntVector dims =
      convertToShape(src.dims, src.w, src.h, src.d, src.c, src.cstep);
  device::TensorDesc tensor_desc(data_type, data_format, dims,
                                 base::SizeVector());
  void *data_ptr = src.data;
  dst->create(device, tensor_desc, data_ptr, name);
  return base::kStatusCodeOk;
}

ncnn::Mat NcnnConvert::matConvertFromTensor(device::Tensor *src) {
  if (!device::isHostDeviceType(src->getDeviceType())) {
    ncnn::Mat dst;
    return dst;
  }
  void *data = src->getPtr();
  int elemsize = src->getDataType().size();
  base::IntVector shape = src->getShape();
  if (shape.size() == 2) {
    ncnn::Mat dst(shape[1], data, elemsize);
    return dst;
  } else if (shape.size() == 3) {
    ncnn::Mat dst(shape[2], shape[1], data, elemsize);
    return dst;
  } else if (shape.size() == 4) {
    ncnn::Mat dst(shape[3], shape[2], shape[1], data, elemsize);
    return dst;
  } else if (shape.size() == 5) {
    ncnn::Mat dst(shape[4], shape[3], shape[2], shape[1], data, elemsize);
    return dst;
  } else {
    ncnn::Mat dst;
    return dst;
  }
}

device::Tensor *NcnnConvert::blobConvertToTensor(ncnn::Blob &src) {
  device::Device *device = device::getDefaultHostDevice();
  base::DataType data_type = convertToDataType(src.shape.elemsize);
  base::DataFormat data_format = convertToDataFormat(
      src.shape.elempack, src.shape.dims, src.shape.w, src.shape.h, src.shape.d,
      src.shape.c, src.shape.cstep);  // 目前只使用了shape.dims
  base::IntVector dims =
      convertToShape(src.shape.dims, src.shape.w, src.shape.h, src.shape.d,
                     src.shape.c, src.shape.cstep);
  device::TensorDesc tensor_desc(data_type, data_format, dims,
                                 base::SizeVector());
  std::string name = src.name;
  device::Tensor *dst = nullptr;
  if (src.shape.data != nullptr) {
    void *data_ptr = src.shape.data;
    base::IntVector memory_config = base::IntVector();
    dst =
        new device::Tensor(device, tensor_desc, data_ptr, name, memory_config);
  } else {
    dst = new device::Tensor(tensor_desc, name);
  }

  return dst;
}

}  // namespace inference
}  // namespace nndeploy
