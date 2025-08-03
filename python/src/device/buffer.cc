
#include "nndeploy/device/buffer.h"

#include "device/tensor_util.h"
#include "nndeploy/device/device.h"
#include "nndeploy/device/memory_pool.h"
#include "nndeploy/device/tensor.h"
#include "nndeploy/device/type.h"
#include "nndeploy/device/util.h"
#include "nndeploy_api_registry.h"

namespace nndeploy {
namespace device {

std::string getPyBufferFormat(base::DataType data_type) {
  std::string format;
  if (data_type.code_ == base::DataTypeCode::kDataTypeCodeUint) {
    if (data_type.bits_ == 8) {
      format = "B";
    } else if (data_type.bits_ == 16) {
      format = "H";
    } else if (data_type.bits_ == 32) {
      format = "I";
    } else if (data_type.bits_ == 64) {
      format = "Q";
    }
  } else if (data_type.code_ == base::DataTypeCode::kDataTypeCodeInt) {
    if (data_type.bits_ == 8) {
      format = "b";
    } else if (data_type.bits_ == 16) {
      format = "h";
    } else if (data_type.bits_ == 32) {
      format = "i";
    } else if (data_type.bits_ == 64) {
      format = "q";
    }
  } else if (data_type.code_ == base::DataTypeCode::kDataTypeCodeFp) {
    if (data_type.bits_ == 16) {
      format = "e";
    } else if (data_type.bits_ == 32) {
      format = "f";
    } else if (data_type.bits_ == 64) {
      format = "d";
    }
  } else {
    format = "";
  }
  return format;
}

base::DataType getDataTypeFromNumpy(char kind, int itemsize) {
  base::DataType data_type;
  int bits = itemsize * 8;
  switch (kind) {
    case 'B':
      data_type = base::DataType(base::DataTypeCode::kDataTypeCodeUint, bits);
      break;
    case 'H':
      data_type = base::DataType(base::DataTypeCode::kDataTypeCodeUint, bits);
      break;
    case 'I':
      data_type = base::DataType(base::DataTypeCode::kDataTypeCodeUint, bits);
    case 'Q':
      data_type = base::DataType(base::DataTypeCode::kDataTypeCodeUint, bits);
      break;
    case 'b':
      data_type = base::DataType(base::DataTypeCode::kDataTypeCodeInt, bits);
      break;
    case 'h':
      data_type = base::DataType(base::DataTypeCode::kDataTypeCodeInt, bits);
      break;
    case 'i':
      data_type = base::DataType(base::DataTypeCode::kDataTypeCodeInt, bits);
      break;
    case 'q':
      data_type = base::DataType(base::DataTypeCode::kDataTypeCodeInt, bits);
      break;
    case 'e':
      data_type = base::DataType(base::DataTypeCode::kDataTypeCodeFp, bits);
      break;
    case 'f':
      data_type = base::DataType(base::DataTypeCode::kDataTypeCodeFp, bits);
      break;
    case 'd':
      data_type = base::DataType(base::DataTypeCode::kDataTypeCodeFp, bits);
      break;
    case 'l':
      data_type = base::DataType(base::DataTypeCode::kDataTypeCodeInt, bits);
      break;
    default:
      std::stringstream ss;
      ss << "convert numpy.ndarray to nndeploy Tensor only support kind = "
            "f, i, u now, but given "
         << kind;
      pybind11::pybind11_fail(ss.str());
  }
  return data_type;
}

py::buffer_info bufferToBufferInfo(device::Buffer *buffer,
                                   const py::dtype &dt) {
  void *data = nullptr;
  if (device::isHostDeviceType(
          buffer->getDeviceType())) {  // 如果是host，则直接将其传递给numpy
                                       // array
    data = buffer->getData();
  } else {
    std::stringstream ss;
    ss << "convert nndeploy.device.Buffer to numpy array only support device "
          ":host but get device_code:"
       << base::deviceTypeToString(buffer->getDeviceType().code_);

    pybind11::pybind11_fail(ss.str());
  }

  if (data == nullptr) {
    std::ostringstream ss;
    ss << "Convert nndeploy Buffer to numpy.ndarray. Get data_ptr==nullptr";

    py::pybind11_fail(ss.str());
  }
  auto elemsize = dt.itemsize();
  //   std::string format = dt.char_() + std::to_string(dt.itemsize());
  std::string format(1, dt.char_());

  auto size_vector = buffer->getSizeVector();
  base::IntVector shape;
  for (auto i = 0; i < size_vector.size(); i++) {
    int32_t size_int = (int32_t)size_vector[i];
    if (i == size_vector.size() - 1) {
      size_int /= elemsize;
    }
    shape.push_back(size_int);
  }
  auto dims = shape.size();
  auto strides = calculateStridesBaseShape(
      shape);  // nndeploy中的strides可能为空，根据shape重新计算
  for (int i = 0; i < strides.size(); i++) {
    strides[i] = strides[i] * elemsize;
  }

  // https://pybind11.readthedocs.io/en/stable/advanced/pycpp/numpy.html#buffer-protocol
  return py::buffer_info(data,     /* Pointer to buffer */
                         elemsize, /* Size of one scalar */
                         format,   /* Python struct-style format descriptor */
                         dims,     /* Number of dimensions */
                         shape,    /* Buffer dimensions */
                         strides   /* Strides (in bytes) for each index */

  );
}

Buffer bufferFromNumpy(const py::array &array) {
  auto shape = array.shape();
  auto dtype = array.dtype();
  auto size = array.size();
  auto strides = array.strides();
  auto data = array.data();
  size_t elemsize = dtype.itemsize();
  size = size * elemsize;
  BufferDesc desc(size);
  Device *device = getDefaultHostDevice();
  return Buffer(device, desc, const_cast<void *>(data));
}

NNDEPLOY_API_PYBIND11_MODULE("device", m) {
  py::class_<Buffer>(m, "Buffer", py::buffer_protocol())
      .def(py::init<Device *, size_t>(), py::arg("device"), py::arg("size"))
      .def(py::init<Device *, const BufferDesc &>(), py::arg("device"),
           py::arg("desc"))
      .def(py::init<Device *, size_t, void *>(), py::arg("device"),
           py::arg("size"), py::arg("ptr"))
      .def(py::init<Device *, const BufferDesc &, void *>(), py::arg("device"),
           py::arg("desc"), py::arg("ptr"))
      .def(py::init<Device *, size_t, void *, base::MemoryType>(),
           py::arg("device"), py::arg("size"), py::arg("ptr"),
           py::arg("memory_type"))
      .def(py::init<Device *, const BufferDesc &, void *, base::MemoryType>(),
           py::arg("device"), py::arg("desc"), py::arg("ptr"),
           py::arg("memory_type"))
      .def(py::init<MemoryPool *, size_t>(), py::arg("memory_pool"),
           py::arg("size"))
      .def(py::init<MemoryPool *, const BufferDesc &>(), py::arg("memory_pool"),
           py::arg("desc"))
      .def(py::init<MemoryPool *, size_t, void *, base::MemoryType>(),
           py::arg("memory_pool"), py::arg("size"), py::arg("ptr"),
           py::arg("memory_type"))
      .def(py::init<MemoryPool *, const BufferDesc &, void *,
                    base::MemoryType>(),
           py::arg("memory_pool"), py::arg("desc"), py::arg("ptr"),
           py::arg("memory_type"))
      .def(py::init<const Buffer &>(), py::arg("buffer"))
      .def("clone", &Buffer::clone, py::return_value_policy::take_ownership,
           "Clone the buffer")
      .def("copy_to", &Buffer::copyTo, py::arg("dst"),
           "Copy the buffer to the destination buffer")
      .def("serialize", py::overload_cast<std::string&>(&Buffer::serialize), py::arg("bin_str"),
           "Serialize the buffer to a binary string")
      .def("deserialize", py::overload_cast<const std::string&>(&Buffer::deserialize), py::arg("bin_str"),
           "Deserialize the buffer from a binary string")
      .def("print",
           [](const Buffer &self) {
             std::ostringstream os;
             self.print(os);
           })
      .def("just_modify",
           py::overload_cast<const size_t &>(&Buffer::justModify),
           py::arg("size"), "Modify the buffer size")
      .def("just_modify",
           py::overload_cast<const base::SizeVector &>(&Buffer::justModify),
           py::arg("size"), "Modify the buffer size")
      .def("just_modify",
           py::overload_cast<const BufferDesc &>(&Buffer::justModify),
           py::arg("desc"), "Modify the buffer descriptor")
      .def("empty", &Buffer::empty, "Check if the buffer is empty")
      .def("get_device_type", &Buffer::getDeviceType,
           "Get the device type of the buffer")
      .def("get_device", &Buffer::getDevice, py::return_value_policy::reference,
           "Get the device of the buffer")
      .def("get_memory_pool", &Buffer::getMemoryPool,
           py::return_value_policy::reference,
           "Get the memory pool of the buffer")
      .def("is_memory_pool", &Buffer::isMemoryPool,
           "Check if the buffer is from a memory pool")
      .def("get_desc", &Buffer::getDesc, "Get the buffer descriptor")
      .def("get_size", &Buffer::getSize, "Get the size of the buffer")
      .def("get_size_vector", &Buffer::getSizeVector,
           "Get the size vector of the buffer")
      .def("get_real_size", &Buffer::getRealSize,
           "Get the real size of the buffer")
      .def("get_real_size_vector", &Buffer::getRealSizeVector,
           "Get the real size vector of the buffer")
      .def("get_config", &Buffer::getConfig,
           "Get the configuration of the buffer")
      .def("get_data", &Buffer::getData, py::return_value_policy::reference,
           "Get the data pointer of the buffer")
      .def("get_memory_type", &Buffer::getMemoryType,
           "Get the memory type of the buffer")
      .def("add_ref", &Buffer::addRef,
           "Increase the reference count of the buffer")
      .def("sub_ref", &Buffer::subRef,
           "Decrease the reference count of the buffer")
      .def("__str__",
           [](const Buffer &self) {
             std::ostringstream os;
             os << "<nndeploy._nndeploy_internal.device.Buffer object at "
                << static_cast<const void *>(&self) << ">";
             self.print(os);
             return os.str();
           })
      .def_buffer([](Buffer &self) {
        return bufferToBufferInfo(&self, py::dtype::of<uint8_t>());
      })
      .def("__array__",
           [](Buffer &self) {
             return bufferToBufferInfo(&self, py::dtype::of<uint8_t>());
           })
      .def(
          "to_numpy_v0",
          [](Buffer &self, const py::dtype &dtype) {
            py::buffer_info buffer_info = bufferToBufferInfo(&self, dtype);
            return py::array(buffer_info);
          },
          py::arg("dtype"),
          "Convert buffer to numpy array with specified dtype")
      .def(
          "to_numpy_v1",
          [](Buffer &self, py::object dtype_obj) {
            py::dtype dtype = py::dtype::from_args(dtype_obj);
            py::buffer_info buffer_info = bufferToBufferInfo(&self, dtype);
            return py::array(buffer_info);
          },
          py::arg("dtype"), "Convert buffer to numpy array with dtype object")
      .def_static("from_numpy", [](const py::array &array) {
        return bufferFromNumpy(array);
      });
}

}  // namespace device
}  // namespace nndeploy
