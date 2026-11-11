#include "device/tensor_util.h"

#include "device/buffer.h"

namespace nndeploy {
namespace device {

py::array tensorArray(Tensor &tensor) {
  auto *buffer = tensor.getBuffer();
  if (!buffer || !isHostDeviceType(buffer->getDeviceType()))
    throw py::value_error("only tensors with host storage can be exported to NumPy");
  auto desc = tensor.getDesc();
  size_t span;
  if (!desc.getStorageSize(span) || span > buffer->getRealSize() ||
      span > static_cast<size_t>(PY_SSIZE_T_MAX) || (span && !buffer->getData()))
    throw py::value_error("tensor layout exceeds its backing storage");
  auto format = getPyBufferFormat(desc.data_type_);
  if (format.empty() || desc.data_type_.lanes_ != 1)
    throw py::type_error("unsupported tensor dtype");
  std::vector<py::ssize_t> strides;
  for (size_t stride : desc.stride_) {
    if (stride > static_cast<size_t>(PY_SSIZE_T_MAX))
      throw py::value_error("tensor stride exceeds Python's size limit");
    strides.push_back(static_cast<py::ssize_t>(stride));
  }
  return py::array(py::dtype(format), desc.shape_, strides, buffer->getData(),
                   py::cast(Buffer(*buffer)));
}

static std::unique_ptr<Tensor> copyTensorToDevice(Tensor &src, Device *dst) {
  auto result = std::make_unique<Tensor>(dst, src.getDesc());
  if (src.getSize()) {
    if (!result->getData() || result->getSize() < src.getSize())
      throw std::runtime_error("failed to allocate destination tensor");
    auto status = src.copyTo(result.get());
    if (status != base::kStatusCodeOk) throw std::runtime_error(status.desc());
  }
  return result;
}

device::Tensor* bufferInfoToTensorByDeviceTypeCode(
    const py::buffer& buffer, const base::DeviceTypeCode& device_type_code) {
  base::DeviceType device_type = device_type_code;
  return bufferInfoToTensor(buffer, device_type);
}

device::Tensor* bufferInfoToTensor(const py::buffer& buffer,
                                   const base::DeviceType& device_type) {

  auto array = py::array::ensure(buffer);
  if (!array) throw py::type_error("expected a NumPy-compatible buffer");
  static const base::DataFormat formats[] = {
      base::kDataFormatNotSupport, base::kDataFormatN, base::kDataFormatNC,
      base::kDataFormatNCL, base::kDataFormatNCHW, base::kDataFormatNCDHW};
  if (array.ndim() > 5) throw py::value_error("tensor rank must be at most 5");
  TensorDesc desc;
  desc.data_format_ = formats[array.ndim()];
  for (py::ssize_t i = 0; i < array.ndim(); ++i) {
    if (array.shape(i) > INT_MAX) throw py::value_error("tensor dimension exceeds INT_MAX");
    desc.shape_.push_back(static_cast<int>(array.shape(i)));
  }
  auto dtype = array.dtype();
  base::DataTypeCode code;
  switch (dtype.kind()) {
    case 'i': code = base::kDataTypeCodeInt; break;
    case 'u': code = base::kDataTypeCodeUint; break;
    case 'f': code = base::kDataTypeCodeFp; break;
    default: throw py::type_error("unsupported NumPy tensor dtype");
  }
  if (dtype.itemsize() > 8) throw py::type_error("unsupported NumPy tensor item size");
  desc.data_type_ = base::DataType(code, dtype.itemsize() * 8);
  if (getPyBufferFormat(desc.data_type_).empty()) throw py::type_error("unsupported tensor dtype");
  if (!dtype.attr("isnative").cast<bool>())
    array = array.attr("astype")(dtype.attr("newbyteorder")("=")).cast<py::array>();

  Device *dst = getDevice(device_type);
  if (!dst) throw py::value_error("requested tensor device is unavailable");
  const bool host = isHostDeviceType(device_type);
  auto storage = std::make_unique<Buffer>(bufferFromNumpy(array, host ? dst : getDefaultHostDevice()));
  auto tensor = std::make_unique<Tensor>(desc);
  tensor->justModify(storage.get(), false);
  storage.release();
  if (host) return tensor.release();

  return copyTensorToDevice(*tensor, dst).release();
}

device::Tensor* moveTensorToDeviceByDeviceTypeCode(
    device::Tensor* tensor, const base::DeviceTypeCode& device_type_code) {
  base::DeviceType device_type = device_type_code;
  return moveTensorToDevice(tensor, device_type);
}

device::Tensor* moveTensorToDevice(device::Tensor* tensor,
                                   const base::DeviceType& device_type) {
  auto *src = tensor->getDevice();
  if (!src) throw py::value_error("source tensor has no device");
  if (src->getDeviceType() == device_type) return tensor;
  auto *dst = getDevice(device_type);
  if (!dst) throw py::value_error("requested tensor device is unavailable");
  return copyTensorToDevice(*tensor, dst).release();
}

}  // namespace device
}  // namespace nndeploy