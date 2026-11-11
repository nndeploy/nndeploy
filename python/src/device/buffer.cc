
#include "nndeploy/device/buffer.h"

#include "device/buffer.h"
#include "numpy_array.h"
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

static void checkRawDtype(const py::dtype &dtype) {
  if (dtype.attr("hasobject").cast<bool>())
    throw py::type_error("raw buffers cannot contain Python object references");
  if (dtype.itemsize() <= 0) throw py::value_error("dtype must have a fixed positive size");
}

static py::array bufferArray(Buffer &buffer, const py::dtype &dtype,
                             py::handle owner = {}) {
  checkRawDtype(dtype);
  if (!isHostDeviceType(buffer.getDeviceType()))
    throw py::value_error("only host buffers can be exported to NumPy");
  auto sizes = buffer.getSizeVector();
  std::vector<py::ssize_t> shape;
  for (size_t size : sizes) {
    if (size > static_cast<size_t>(PY_SSIZE_T_MAX))
      throw py::value_error("buffer dimension exceeds Python's size limit");
    shape.push_back(static_cast<py::ssize_t>(size));
  }
  if (shape.empty()) shape.push_back(0);
  if (shape.back() % dtype.itemsize() != 0)
    throw py::value_error("buffer size is not divisible by dtype itemsize");
  shape.back() /= dtype.itemsize();
  if (buffer.getSize() && !buffer.getData()) throw py::value_error("buffer has no storage");
  return py::array(dtype, shape, buffer.getData(), owner);
}

Buffer bufferFromNumpy(const py::array &array, Device *device) {
  checkRawDtype(array.dtype());
  auto storage = writableContiguousArray(array);
  if (!device) throw py::value_error("host device is unavailable");
  const size_t bytes = static_cast<size_t>(storage.nbytes());
  void *ptr = storage.mutable_data();
  // Keep the normalized ndarray (and its base exporter) with every C++ alias.
  std::shared_ptr<void> data(ptr, [owner = storage.release().ptr()](void *) {
    py::gil_scoped_acquire acquire;
    Py_DECREF(owner);
  });
  return Buffer(device, BufferDesc(bytes), std::move(data));
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
      .def("serialize", [](Buffer &self) {
        std::string data;
        auto status = self.serialize(data);
        if (status != base::kStatusCodeOk) throw std::runtime_error(status.desc());
        return py::bytes(data);
      })
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
        return bufferArray(self, py::dtype::of<uint8_t>(), py::cast(Buffer(self))).request();
      })
      .def("__array__", [](Buffer &self) {
        return bufferArray(self, py::dtype::of<uint8_t>(), py::cast(Buffer(self)));
      })
      .def("to_numpy_v0", [](Buffer &self, const py::dtype &dtype) {
        return bufferArray(self, dtype);
      }, py::arg("dtype"))
      .def("to_numpy_v1", [](Buffer &self, py::object dtype) {
        return bufferArray(self, py::dtype::from_args(dtype));
      }, py::arg("dtype"))
      .def_static("from_numpy", [](const py::array &array) {
        return bufferFromNumpy(array, getDefaultHostDevice());
      });
}

}  // namespace device
}  // namespace nndeploy
