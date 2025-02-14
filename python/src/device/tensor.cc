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
      .def("clone", &Tensor::clone, "Clone the tensor")
      .def("copyTo", &Tensor::copyTo, py::arg("dst"),
           "Copy the tensor to the destination tensor")
      .def("print",
           [](const Tensor &self) {
             std::ostringstream os;
             self.print(os);
           })
      .def("reshape", &Tensor::reshape, py::arg("shape"), "Reshape the tensor")
      .def("justModify",
           py::overload_cast<const TensorDesc &>(&Tensor::justModify),
           py::arg("desc"), "Modify the tensor descriptor")
      .def("justModify", py::overload_cast<Buffer *, bool>(&Tensor::justModify),
           py::arg("buffer"), py::arg("is_external") = true,
           "Modify the tensor buffer")
      .def("empty", &Tensor::empty, "Check if the tensor is empty")
      .def("isContinue", &Tensor::isContinue,
           "Check if the tensor data is continuous")
      .def("isExternalBuffer", &Tensor::isExternalBuffer,
           "Check if the tensor buffer is external")
      .def("getName", &Tensor::getName, "Get the name of the tensor")
      .def("setName", &Tensor::setName, py::arg("name"),
           "Set the name of the tensor")
      .def("getDesc", &Tensor::getDesc, "Get the tensor descriptor")
      .def("getDataType", &Tensor::getDataType,
           "Get the data type of the tensor")
      .def("setDataType", &Tensor::setDataType, py::arg("data_type"),
           "Set the data type of the tensor")
      .def("getDataFormat", &Tensor::getDataFormat,
           "Get the data format of the tensor")
      .def("setDataFormat", &Tensor::setDataFormat, py::arg("data_format"),
           "Set the data format of the tensor")
      .def("getShape", &Tensor::getShape, "Get the shape of the tensor")
      .def("getShapeIndex", &Tensor::getShapeIndex, py::arg("index"),
           "Get the shape value at the given index")
      .def("getBatch", &Tensor::getBatch, "Get the batch size of the tensor")
      .def("getChannel", &Tensor::getChannel,
           "Get the channel size of the tensor")
      .def("getDepth", &Tensor::getDepth, "Get the depth of the tensor")
      .def("getHeight", &Tensor::getHeight, "Get the height of the tensor")
      .def("getWidth", &Tensor::getWidth, "Get the width of the tensor")
      .def("getStride", &Tensor::getStride, "Get the stride of the tensor")
      .def("getStrideIndex", &Tensor::getStrideIndex, py::arg("index"),
           "Get the stride value at the given index")
      .def("getBuffer", &Tensor::getBuffer, py::return_value_policy::reference,
           "Get the buffer of the tensor")
      .def("getDeviceType", &Tensor::getDeviceType,
           "Get the device type of the tensor")
      .def("getDevice", &Tensor::getDevice, py::return_value_policy::reference,
           "Get the device of the tensor")
      .def("getMemoryPool", &Tensor::getMemoryPool,
           py::return_value_policy::reference,
           "Get the memory pool of the tensor")
      .def("isMemoryPool", &Tensor::isMemoryPool,
           "Check if the tensor is from a memory pool")
      .def("getBufferDesc", &Tensor::getBufferDesc,
           "Get the buffer descriptor of the tensor")
      .def("getSize", &Tensor::getSize, "Get the size of the tensor")
      .def("getSizeVector", &Tensor::getSizeVector,
           "Get the size vector of the tensor")
      .def("getRealSize", &Tensor::getRealSize,
           "Get the real size of the tensor")
      .def("getRealSizeVector", &Tensor::getRealSizeVector,
           "Get the real size vector of the tensor")
      .def("getConfig", &Tensor::getConfig,
           "Get the configuration of the tensor")
      .def("getData", &Tensor::getData, py::return_value_policy::reference,
           "Get the data pointer of the tensor")
      .def("getMemoryType", &Tensor::getMemoryType,
           "Get the memory type of the tensor")
      .def("addRef", &Tensor::addRef,
           "Increase the reference count of the tensor")
      .def("subRef", &Tensor::subRef,
           "Decrease the reference count of the tensor")
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
      .def("to", [](Tensor &self, const base::DeviceType &device_type) {
        return moveTensorToDevice(&self, device_type.code_);
      });

  m.def("tensor_to_numpy", [](Tensor &self) {
    py::buffer_info buffer_info = tensorToBufferInfo(&self);
    return py::array(buffer_info);
  });

  m.def("tensor_from_numpy",
        [](const py::buffer &buffer, const base::DeviceType &device_type) {
          return bufferInfoToTensor(buffer, device_type);
        });
}

}  // namespace device
}  // namespace nndeploy