#include "nndeploy/device/tensor.h"

#include <pybind11/stl.h>

#include "device/tensor_util.h"
#include "nndeploy/device/type.h"
#include "nndeploy_api_registry.h"

namespace nndeploy {
namespace device {

NNDEPLOY_API_PYBIND11_MODULE("device", m) {
  // py::class_<device::Tensor>(m, "Tensor", py::buffer_protocol())
  //     .def(py::init<>())
  //     .def(py::init<const std::string &>())
  //     .def(py::init<const device::TensorDesc &, const std::string &>(),
  //          py::arg("desc"),
  //          py::arg("name") = "")
  //     .def_buffer(
  //         tensorToBufferInfo)  // 通过np.asarray(tensor)转换为numpy array
  //     .def(py::init([](py::buffer const b, base::DeviceTypeCode device_code)
  //     {
  //       return bufferInfoToTensor(b, device_code);
  //     }))
  //     .def("copyTo", &Tensor::copyTo)
  //     .def("getName", &Tensor::getName)
  //     // 移动Tensor到其他设备上
  //     .def(
  //         "to",
  //         [](py::object self, base::DeviceTypeCode device_code) {
  //           device::Tensor *tensor = self.cast<device::Tensor *>();
  //           return moveTensorToDevice(tensor, device_code);
  //         },
  //         py::return_value_policy::reference)
  //     .def_property_readonly("shape", &Tensor::getShape);

  py::class_<Tensor>(m, "Tensor", py::buffer_protocol())
      .def(py::init<>())
      .def(py::init<const std::string &>(), py::arg("name"))
      .def(py::init<const TensorDesc &, const std::string &>(), py::arg("desc"),
           py::arg("name") = "")
      .def(py::init<const TensorDesc &, Buffer *, const std::string &>(),
           py::arg("desc"), py::arg("buffer"), py::arg("name") = "")
      .def(py::init<Device *, const TensorDesc &, const std::string &,
                    const base::IntVector &>(),
           py::arg("device"), py::arg("desc"), py::arg("name") = "",
           py::arg("config") = base::IntVector())
      .def(py::init<Device *, const TensorDesc &, void *, const std::string &,
                    const base::IntVector &>(),
           py::arg("device"), py::arg("desc"), py::arg("data_ptr"),
           py::arg("name") = "", py::arg("config") = base::IntVector())
      .def(py::init<MemoryPool *, const TensorDesc &, const std::string &,
                    const base::IntVector &>(),
           py::arg("memory_pool"), py::arg("desc"), py::arg("name") = "",
           py::arg("config") = base::IntVector())
      .def(py::init<MemoryPool *, const TensorDesc &, void *,
                    const std::string &, const base::IntVector &>(),
           py::arg("memory_pool"), py::arg("desc"), py::arg("data_ptr"),
           py::arg("name") = "", py::arg("config") = base::IntVector())
      .def(py::init<const Tensor &>(), py::arg("tensor"))
      .def("create", py::overload_cast<const std::string &>(&Tensor::create),
           py::arg("name"))
      .def("create",
           py::overload_cast<const TensorDesc &, const std::string &>(
               &Tensor::create),
           py::arg("desc"), py::arg("name") = "")
      .def("create",
           py::overload_cast<const TensorDesc &, Buffer *, const std::string &>(
               &Tensor::create),
           py::arg("desc"), py::arg("buffer"), py::arg("name") = "")
      .def("create",
           py::overload_cast<Device *, const TensorDesc &, const std::string &,
                             const base::IntVector &>(&Tensor::create),
           py::arg("device"), py::arg("desc"), py::arg("name") = "",
           py::arg("config") = base::IntVector())
      .def("create",
           py::overload_cast<Device *, const TensorDesc &, void *,
                             const std::string &, const base::IntVector &>(
               &Tensor::create),
           py::arg("device"), py::arg("desc"), py::arg("data_ptr"),
           py::arg("name") = "", py::arg("config") = base::IntVector())
      .def("create",
           py::overload_cast<MemoryPool *, const TensorDesc &,
                             const std::string &, const base::IntVector &>(
               &Tensor::create),
           py::arg("memory_pool"), py::arg("desc"), py::arg("name") = "",
           py::arg("config") = base::IntVector())
      .def("create",
           py::overload_cast<MemoryPool *, const TensorDesc &, void *,
                             const std::string &, const base::IntVector &>(
               &Tensor::create),
           py::arg("memory_pool"), py::arg("desc"), py::arg("data_ptr"),
           py::arg("name") = "", py::arg("config") = base::IntVector())
      .def("clear", &Tensor::clear)
      .def("allocate",
           py::overload_cast<Device *, const base::IntVector &>(
               &Tensor::allocate),
           py::arg("device"), py::arg("config") = base::IntVector())
      .def("allocate",
           py::overload_cast<MemoryPool *, const base::IntVector &>(
               &Tensor::allocate),
           py::arg("memory_pool"), py::arg("config") = base::IntVector())
      .def("deallocate", &Tensor::deallocate)
      .def(
          "set",
          [](Tensor &self, py::object value) {
            if (py::isinstance<py::int_>(value)) {
              return self.set(value.cast<int>());
            } else if (py::isinstance<py::float_>(value)) {
              return self.set(value.cast<float>());
            } else {
              throw py::type_error("Unsupported type for Tensor::set");
            }
          },
          py::arg("value"))
      .def("reshape", &Tensor::reshape, py::arg("shape"))
      .def("just_modify",
           py::overload_cast<const TensorDesc &>(&Tensor::justModify),
           py::arg("desc"))
      .def("just_modify",
           py::overload_cast<Buffer *, bool>(&Tensor::justModify),
           py::arg("buffer"), py::arg("is_external") = true)
      .def("clone", &Tensor::clone, py::return_value_policy::take_ownership)
      .def("copy_to", &Tensor::copyTo, py::arg("dst"))
      .def(
          "serialize",
          [](Tensor &self, py::object &stream) {
            std::ostream os(stream.cast<std::streambuf *>());
            return self.serialize(os);
          },
          py::arg("stream"))
      .def(
          "deserialize",
          [](Tensor &self, py::object &stream) {
            std::istream is(stream.cast<std::streambuf *>());
            return self.deserialize(is);
          },
          py::arg("stream"))
      .def(
          "print",
          [](const Tensor &self, py::object &stream) {
            std::ostream os(stream.cast<std::streambuf *>());
            self.print(os);
          },
          py::arg("stream") = py::none())
      .def("is_same_device", &Tensor::isSameDevice, py::arg("tensor"))
      .def("is_same_memory_pool", &Tensor::isSameMemoryPool, py::arg("tensor"))
      .def("is_same_desc", &Tensor::isSameDesc, py::arg("tensor"))
      .def("empty", &Tensor::empty)
      .def("is_continue", &Tensor::isContinue)
      .def("is_external_buffer", &Tensor::isExternalBuffer)
      .def("get_name", &Tensor::getName)
      .def("set_name", &Tensor::setName, py::arg("name"))
      .def("get_desc", &Tensor::getDesc)
      .def("get_data_type", &Tensor::getDataType)
      .def("set_data_type", &Tensor::setDataType, py::arg("data_type"))
      .def("get_data_format", &Tensor::getDataFormat)
      .def("set_data_format", &Tensor::setDataFormat, py::arg("data_format"))
      .def("get_shape", &Tensor::getShape)
      .def_property_readonly("shape", &Tensor::getShape)
      .def("get_shape_index", &Tensor::getShapeIndex, py::arg("index"))
      .def("get_batch", &Tensor::getBatch)
      .def("get_channel", &Tensor::getChannel)
      .def("get_depth", &Tensor::getDepth)
      .def("get_height", &Tensor::getHeight)
      .def("get_width", &Tensor::getWidth)
      .def("get_stride", &Tensor::getStride)
      .def("get_stride_index", &Tensor::getStrideIndex, py::arg("index"))
      .def("get_buffer", &Tensor::getBuffer, py::return_value_policy::reference)
      .def("get_device_type", &Tensor::getDeviceType)
      .def("get_device", &Tensor::getDevice, py::return_value_policy::reference)
      .def("get_memory_pool", &Tensor::getMemoryPool,
           py::return_value_policy::reference)
      .def("is_memory_pool", &Tensor::isMemoryPool)
      .def("get_buffer_desc", &Tensor::getBufferDesc)
      .def("get_size", &Tensor::getSize)
      .def("get_size_vector", &Tensor::getSizeVector)
      .def("get_real_size", &Tensor::getRealSize)
      .def("get_real_size_vector", &Tensor::getRealSizeVector)
      .def("get_config", &Tensor::getConfig)
      .def("get_data", &Tensor::getData, py::return_value_policy::reference)
      .def("get_memory_type", &Tensor::getMemoryType)
      .def("add_ref", &Tensor::addRef)
      .def("sub_ref", &Tensor::subRef)
      .def("__str__",
           [](const Tensor &self) {
             std::ostringstream os;
             os << "<nndeploy._nndeploy_internal.device.Tensor object at "
                << static_cast<const void *>(&self) << ">";
             self.print(os);
             return os.str();
           })
      .def_buffer([](Tensor &self) { return tensorToBufferInfo(&self); })
      .def("__array__", [](Tensor &self) { return tensorToBufferInfo(&self); })
      .def("to",
           [](Tensor &self, const base::DeviceType &device_type) {
             return moveTensorToDevice(&self, device_type);
           })
      .def("to_numpy",
           [](Tensor &self) {
             py::buffer_info buffer_info = tensorToBufferInfo(&self);
             return py::array(buffer_info);
           })
      .def_static("from_numpy", [](const py::buffer &buffer,
                                   const base::DeviceType &device_type) {
        return bufferInfoToTensor(buffer, device_type);
      });

  m.def("create_tensor", [](base::TensorType type = base::kTensorTypeDefault) {
    return createTensor(type);
  }, py::return_value_policy::take_ownership);  

}

}  // namespace device
}  // namespace nndeploy