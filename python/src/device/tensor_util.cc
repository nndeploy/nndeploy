#include "device/tensor_util.h"

std::string getTensorFormat(device::Tensor* tensor) {
  std::string format;
  auto elemsize = tensor->getDataType().bits_ / 8;
  if (elemsize == 4) {
    format = pybind11::format_descriptor<float>::format();
  }
  if (elemsize == 2) {
    // see https://docs.python.org/3/library/struct.html#format-characters
    format = "e";
  }
  if (elemsize == 1) {
    format = pybind11::format_descriptor<int8_t>::format();
  }
  return format;
}

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
  void* data = nullptr;
  auto device_type_code = tensor->getDeviceType().code_;
  if (device_type_code ==
      base::kDeviceTypeCodeCpu) {  // 如果是cpu，则直接将其传递给numpy array
    data = tensor->getBuffer()->getData();

  } else {
    std::stringstream ss;
    ss << "convert nndeploy.device.Tensor to numpy array only support device "
          ":cpu  but get device_code:"
       << base::deviceTypeToString(device_type_code);
    pybind11::pybind11_fail(ss.str());
  }

  if (data == nullptr) {
    std::ostringstream ss;
    ss << "Convert nndeploy Tensor to numpy.ndarray. Get data_ptr==nullptr";

    py::pybind11_fail(ss.str());
  }
  auto elemsize = tensor->getDataType().bits_ / 8;
  auto format = getTensorFormat(tensor);
  auto dims = tensor->getShape().size();
  auto strides = calculateStridesBaseShape(
      tensor->getShape());  // nndeploy中的strides可能为空，根据shape重新计算
  for (int i = 0; i < strides.size(); i++) {
    strides[i] = strides[i] * elemsize;
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

device::Tensor* bufferInfoToTensor(py::buffer const b,
                                   base::DeviceTypeCode device_code) {
  device::Tensor* tensor = nullptr;

  py::buffer_info info = b.request();
  if (info.ndim > 4) {
    std::stringstream ss;
    ss << "convert numpy.ndarray to nndeploy Tensor only dims <=4 support now, "
          "but given "
       << info.ndim;
    pybind11::pybind11_fail(ss.str());
  }

  device::TensorDesc desc;

  // 根据numpy中元素的空间大小，赋值一个默认数值类型
  switch (info.itemsize) {
    case 4:
      desc.data_type_ = base::dataTypeOf<float>();
      break;
    case 2:
      desc.data_type_ = base::dataTypeOf<int16_t>();
      break;
    case 1:
      desc.data_type_ = base::dataTypeOf<int8_t>();
      break;
    default:
      std::stringstream ss;
      ss << "convert numpy.ndarray to nndeploy Tensor only support itemsize = "
            "4, 2, 1 "
            "now, "
            "but given "
         << info.itemsize;
      pybind11::pybind11_fail(ss.str());
  }

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
    default:
      desc.data_format_ = base::kDataFormatNotSupport;
      break;
  }

  desc.shape_ = std::vector<int>(info.shape.begin(), info.shape.end());
  void* data_ptr = info.ptr;

  if (device_code == base::kDeviceTypeCodeCpu) {
    auto device = device::getDevice(device_code);
    tensor = new device::Tensor(device, desc, data_ptr);
  } else {
    std::stringstream ss;
    ss << "convert numpy.ndarray to nndeploy Tensor only support device :cpu ";
    pybind11::pybind11_fail(ss.str());
  }
  return tensor;
}

device::Tensor* moveTensorToDevice(device::Tensor* tensor,
                                   base::DeviceTypeCode device_code) {
  auto cur_device = tensor->getDevice();
  auto dst_device = device::getDevice(base::DeviceType(device_code));
  // TODO: 直接对比两个指针表示是同一设备 对不对？
  if (cur_device != dst_device) {
    auto dst_buffer_desc = tensor->getBuffer()->getDesc();
    auto cur_buffer = tensor->getBuffer();
    auto dst_buffer = new device::Buffer(dst_device, dst_buffer_desc);
    if (cur_buffer->copyTo(dst_buffer) != base::kStatusCodeOk) {
      std::stringstream ss;
      ss << "move Tensor from "
         << base::deviceTypeToString(cur_device->getDeviceType()) << " to "
         << base::deviceTypeToString(dst_device->getDeviceType()) << " failed!";
      pybind11::pybind11_fail(ss.str());
    }

    tensor->justModify(dst_buffer);
  }

  return tensor;
}