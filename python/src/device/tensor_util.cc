#include "device/tensor_util.h"

#include "device/buffer.h"

namespace nndeploy {
namespace device {

std::vector<long> calculateStridesBaseShape(const base::IntVector& shape) {
  std::vector<long> strides(shape.size());
  long total_size = 1;

  // 从后往前计算每个维度的 stride
  for (int i = shape.size() - 1; i >= 0; --i) {
    strides[i] = total_size;
    total_size *= shape[i];
  }

  return strides;
}

py::buffer_info tensorToBufferInfo(device::Tensor* tensor) {
  base::DeviceType host_device_type = getDefaultHostDeviceType();
  device::Tensor* host_tensor = nullptr;
  base::DeviceType device_type = tensor->getDeviceType();
  if (device::isHostDeviceType(
          device_type)) {  // 如果是host，则直接将其传递给numpy
                           // array
    host_tensor = tensor;
  } else {
    std::stringstream ss;
    ss << "convert nndeploy.device.Tensor to numpy array only support device"
          ":host  but get device_code:"
       << base::deviceTypeToString(device_type.code_);

    pybind11::pybind11_fail(ss.str());
    // 无法做内存管理
    // host_tensor = moveTensorToDevice(tensor, host_device_type);
  }
  void* data = host_tensor->getBuffer()->getData();
  if (data == nullptr) {
    std::stringstream ss;
    ss << "Convert nndeploy Tensor to numpy.ndarray. Get data_ptr==nullptr";

    py::pybind11_fail(ss.str());
  }
  auto elemsize = tensor->getDataType().size();
  auto format = getPyBufferFormat(tensor->getDataType());
  auto dims = tensor->getShape().size();
  std::vector<long> strides;
  for (auto stride : tensor->getStride()) {
    strides.push_back((long)stride);
  }
  if (strides.empty()) {
    strides = calculateStridesBaseShape(
        tensor->getShape());  // nndeploy中的strides可能为空，根据shape重新计算
    for (int i = 0; i < strides.size(); i++) {
      strides[i] = strides[i] * elemsize;
    }
  }

  // https://pybind11.readthedocs.io/en/stable/advanced/pycpp/numpy.html#buffer-protocol
  return py::buffer_info(data,     /* Pointer to buffer */
                         elemsize, /* Size of one scalar */
                         format,   /* Python struct-style format descriptor */
                         dims,     /* Number of dimensions */
                         tensor->getShape(), /* Buffer dimensions */
                         strides /* Strides (in bytes) for each index */

  );
}

device::Tensor* bufferInfoToTensorByDeviceTypeCode(
    const py::buffer& buffer, const base::DeviceTypeCode& device_type_code) {
  base::DeviceType device_type = device_type_code;
  return bufferInfoToTensor(buffer, device_type);
}

device::Tensor* bufferInfoToTensor(const py::buffer& buffer,
                                   const base::DeviceType& device_type) {
  device::Tensor* tensor = nullptr;

  py::buffer_info info = buffer.request();
  if (info.ndim > 5) {
    std::stringstream ss;
    ss << "convert numpy.ndarray to nndeploy Tensor only dims <=4 support now, "
          "but given "
       << info.ndim;
    pybind11::pybind11_fail(ss.str());
  }

  device::TensorDesc desc;

  // 根据numpy中元素的空间大小，赋值一个默认数值类型
  // base::DataType data_type;
  char kind = info.format.front();
  // int bits = info.itemsize * 8;
  // switch (kind) {
  //   case 'f':
  //     data_type = base::DataType(base::DataTypeCode::kDataTypeCodeFp, bits);
  //     break;
  //   case 'i':
  //     data_type = base::DataType(base::DataTypeCode::kDataTypeCodeInt, bits);
  //     break;
  //   case 'u':
  //     data_type = base::DataType(base::DataTypeCode::kDataTypeCodeUint,
  //     bits); break;
  //   default:
  //     std::stringstream ss;
  //     ss << "convert numpy.ndarray to nndeploy Tensor only support kind = "
  //           "f, i, u "
  //           "now, "
  //           "but given "
  //        << kind;
  //     pybind11::pybind11_fail(ss.str());
  // }
  desc.data_type_ = getDataTypeFromNumpy(kind, info.itemsize);

  // 根据numpy中维度的大小，赋值一个默认格式
  switch (info.ndim) {
    case 1:
      desc.data_format_ = base::kDataFormatN;
      break;
    case 2:
      desc.data_format_ = base::kDataFormatNC;
      break;
    case 3:
      desc.data_format_ = base::kDataFormatNCL;
      break;
    case 4:
      desc.data_format_ = base::kDataFormatNCHW;
      break;
    case 5:
      desc.data_format_ = base::kDataFormatNCDHW;
      break;
    default:
      desc.data_format_ = base::kDataFormatNotSupport;
      break;
  }

  desc.shape_ = std::vector<int>(info.shape.begin(), info.shape.end());
  void* data_ptr = info.ptr;

  if (device::isHostDeviceType(device_type)) {
    auto host_device = device::getDevice(device_type);
    device::Tensor* host_tensor =
        new device::Tensor(host_device, desc, data_ptr);
    if (host_tensor == nullptr) {
      return nullptr;
    }
    auto dst_tensor = host_tensor;
    return dst_tensor;
  } else {
    auto host_device = device::getDefaultHostDevice();
    device::Tensor* host_tensor =
        new device::Tensor(host_device, desc, data_ptr);
    if (host_tensor == nullptr) {
      return nullptr;
    }
    auto dst_device = device::getDevice(device_type);
    auto cur_buffer = host_tensor->getBuffer();

    auto dst_tensor = new device::Tensor(dst_device, desc);
    if (cur_buffer->copyTo(dst_tensor->getBuffer()) != base::kStatusCodeOk) {
      // std::stringstream ss;
      // ss << "move Tensor from "
      //    << base::deviceTypeToString(cur_device->getDeviceType()) << " to "
      //    << base::deviceTypeToString(dst_device->getDeviceType()) << "
      //    failed!";
      // pybind11::pybind11_fail(ss.str());
      return nullptr;
    }

    delete host_tensor;

    return dst_tensor;
  }
}

device::Tensor* moveTensorToDeviceByDeviceTypeCode(
    device::Tensor* tensor, const base::DeviceTypeCode& device_type_code) {
  base::DeviceType device_type = device_type_code;
  return moveTensorToDevice(tensor, device_type);
}

device::Tensor* moveTensorToDevice(device::Tensor* tensor,
                                   const base::DeviceType& device_type) {
  auto cur_device = tensor->getDevice();
  auto dst_device = device::getDevice(device_type);
  if (cur_device->getDeviceType() != device_type) {
    auto cur_desc = tensor->getDesc();
    auto cur_buffer = tensor->getBuffer();

    auto dst_tensor = new device::Tensor(dst_device, cur_desc);
    if (cur_buffer->copyTo(dst_tensor->getBuffer()) != base::kStatusCodeOk) {
      std::stringstream ss;
      ss << "move Tensor from "
         << base::deviceTypeToString(cur_device->getDeviceType()) << " to "
         << base::deviceTypeToString(dst_device->getDeviceType()) << " failed!";
      pybind11::pybind11_fail(ss.str());
    }
    return dst_tensor;
  }

  return tensor;
}

}  // namespace device
}  // namespace nndeploy