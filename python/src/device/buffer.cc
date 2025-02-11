
#include "nndeploy/device/buffer.h"

#include "nndeploy/device/device.h"
#include "nndeploy/device/memory_pool.h"
#include "nndeploy/device/tensor.h"
#include "nndeploy/device/type.h"
#include "nndeploy/device/util.h"
#include "nndeploy_api_registry.h"

namespace nndeploy {
namespace device {

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
      .def("clone", &Buffer::clone, "Clone the buffer")
      .def("copyTo", &Buffer::copyTo, py::arg("dst"),
           "Copy the buffer to the destination buffer")
      // .def("serialize", &Buffer::serialize, py::arg("stream"),
      //      "Serialize the buffer to a binary stream")
      // .def("deserialize", &Buffer::deserialize, py::arg("stream"),
      //      "Deserialize the buffer from a binary stream")
      .def("print",
           [](const Buffer &self) {
             std::ostringstream os;
             self.print(os);
           })
      .def("justModify", py::overload_cast<const size_t &>(&Buffer::justModify),
           py::arg("size"), "Modify the buffer size")
      .def("justModify",
           py::overload_cast<const base::SizeVector &>(&Buffer::justModify),
           py::arg("size"), "Modify the buffer size")
      .def("justModify",
           py::overload_cast<const BufferDesc &>(&Buffer::justModify),
           py::arg("desc"), "Modify the buffer descriptor")
      .def("empty", &Buffer::empty, "Check if the buffer is empty")
      .def("getDeviceType", &Buffer::getDeviceType,
           "Get the device type of the buffer")
      .def("getDevice", &Buffer::getDevice, py::return_value_policy::reference,
           "Get the device of the buffer")
      .def("getMemoryPool", &Buffer::getMemoryPool,
           py::return_value_policy::reference,
           "Get the memory pool of the buffer")
      .def("isMemoryPool", &Buffer::isMemoryPool,
           "Check if the buffer is from a memory pool")
      .def("getDesc", &Buffer::getDesc, "Get the buffer descriptor")
      .def("getSize", &Buffer::getSize, "Get the size of the buffer")
      .def("getSizeVector", &Buffer::getSizeVector,
           "Get the size vector of the buffer")
      .def("getRealSize", &Buffer::getRealSize,
           "Get the real size of the buffer")
      .def("getRealSizeVector", &Buffer::getRealSizeVector,
           "Get the real size vector of the buffer")
      .def("getConfig", &Buffer::getConfig,
           "Get the configuration of the buffer")
      .def("getData", &Buffer::getData, py::return_value_policy::reference,
           "Get the data pointer of the buffer")
      .def("getMemoryType", &Buffer::getMemoryType,
           "Get the memory type of the buffer")
      .def("addRef", &Buffer::addRef,
           "Increase the reference count of the buffer")
      .def("subRef", &Buffer::subRef,
           "Decrease the reference count of the buffer")
      .def("__str__", [](const Buffer &self) {
        std::ostringstream os;
        os << "<nndeploy._nndeploy_internal.device.Buffer object at "
           << static_cast<const void *>(&self) << ">";
        self.print(os);
        return os.str();
      });
  // .def_buffer([](Buffer &self) {
  //   // 获取buffer的描述信息
  //   BufferDesc desc = self.getDesc();
  //   // 获取buffer的数据指针
  //   void *ptr = self.getData();
  //   // 获取数据类型
  //   py::dtype dt = py::dtype(py::format_descriptor<uint8_t>::format());
  //   switch (desc.getConfig().getDataType()) {
  //     case base::DataType::kInt8:
  //       dt = py::dtype::of<int8_t>();
  //       break;
  //     case base::DataType::kUInt8:
  //       dt = py::dtype::of<uint8_t>();
  //       break;
  //     case base::DataType::kInt32:
  //       dt = py::dtype::of<int32_t>();
  //       break;
  //     case base::DataType::kInt64:
  //       dt = py::dtype::of<int64_t>();
  //       break;
  //     case base::DataType::kFloat16:
  //       dt = py::dtype("e");  // float16
  //       break;
  //     case base::DataType::kFloat32:
  //       dt = py::dtype::of<float>();
  //       break;
  //     case base::DataType::kFloat64:
  //       dt = py::dtype::of<double>();
  //       break;
  //     default:
  //       NNDEPLOY_THROW("Unsupported data type");
  //   }
  //   // 根据buffer的描述信息构造numpy array所需的buffer_info
  //   py::buffer_info info(
  //       ptr,                                    // 数据指针
  //       dt.itemsize(),                          // 元素字节数
  //       dt.format(),                            // 元素格式
  //       desc.getSizeVector().size(),            // 维度数
  //       desc.getSizeVector(),                   // 各维度大小
  //       calculateStridesBaseShape(desc.getSizeVector())  // 各维度步长
  //   );
  //   return info;
  // });
}

}  // namespace device
}  // namespace nndeploy
