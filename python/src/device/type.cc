
#include "nndeploy/device/type.h"

#include "nndeploy_api_registry.h"

namespace nndeploy {
namespace device {

NNDEPLOY_API_PYBIND11_MODULE("device", m) {
  // nndeploy::device::BufferDesc export as device.BufferDesc
  py::class_<BufferDesc>(m, "BufferDesc")
      .def(py::init<>())
      .def(py::init<size_t>())
      .def(py::init<size_t *, size_t>())
      .def(py::init<const base::SizeVector &>())
      .def(py::init<size_t, const base::IntVector &>())
      .def(py::init<const base::SizeVector &, const base::IntVector &>())
      .def(py::init<size_t *, size_t, const base::IntVector &>())
      .def(py::init<const BufferDesc &>())
      .def("__eq__", &BufferDesc::operator==)
      .def("__ne__", &BufferDesc::operator!=)
      .def("__ge__", &BufferDesc::operator>=)
      .def("getSize", &BufferDesc::getSize)
      .def("getSizeVector", &BufferDesc::getSizeVector)
      .def("getRealSize", &BufferDesc::getRealSize)
      .def("getRealSizeVector", &BufferDesc::getRealSizeVector)
      .def("getConfig", &BufferDesc::getConfig)
      .def("isSameConfig", &BufferDesc::isSameConfig)
      .def("isSameDim", &BufferDesc::isSameDim)
      .def("is1D", &BufferDesc::is1D)
      .def("print", &BufferDesc::print)
      .def("justModify",
           py::overload_cast<const size_t &>(&BufferDesc::justModify))
      .def("justModify",
           py::overload_cast<const base::SizeVector &>(&BufferDesc::justModify))
      .def("justModify",
           py::overload_cast<const BufferDesc &>(&BufferDesc::justModify))
      .def("clear", &BufferDesc::clear)
      .def("__str__", [](const BufferDesc &self) {
        std::ostringstream os;
        os << "<nndeploy._nndeploy_internal.device.BufferDesc object at "
           << static_cast<const void *>(&self) << ">";
        self.print(os);
        return os.str();
      });

  // nndeploy::device::TensorDesc export as device.TensorDesc
  py::class_<TensorDesc>(m, "TensorDesc")
      .def(py::init<>())
      .def(
          py::init<base::DataType, base::DataFormat, const base::IntVector &>(),
          py::arg("data_type"), py::arg("format"), py::arg("shape"))
      .def(py::init<base::DataType, base::DataFormat, const base::IntVector &,
                    const base::SizeVector &>(),
           py::arg("data_type"), py::arg("format"), py::arg("shape"),
           py::arg("stride"))
      .def(py::init<const TensorDesc &>(), py::arg("desc"))
      .def("__eq__", &TensorDesc::operator==, py::arg("other"))
      .def("__ne__", &TensorDesc::operator!=, py::arg("other"))
      .def(
          "serialize",
          [](TensorDesc &self, py::object &stream) {
            py::gil_scoped_acquire acquire;
            auto buffer = py::reinterpret_steal<py::object>(
                PyObject_CallMethod(stream.ptr(), "getvalue", nullptr));
            py::gil_scoped_release release;

            std::string str = py::str(buffer);
            std::istringstream iss(str);
            std::ostringstream oss;
            oss << iss.rdbuf();
            return self.serialize(oss);
          },
          py::arg("stream"), "Serialize the tensor desc to a binary stream")
      .def(
          "deserialize",
          [](TensorDesc &self, py::object &stream) {
            py::gil_scoped_acquire acquire;
            auto buffer = py::reinterpret_steal<py::object>(
                PyObject_CallMethod(stream.ptr(), "getvalue", nullptr));
            py::gil_scoped_release release;

            std::string str = py::str(buffer);
            std::istringstream iss(str);
            return self.deserialize(iss);
          },
          py::arg("stream"), "Deserialize the tensor desc from a binary stream")
      .def("print", &TensorDesc::print, py::arg("stream"))
      .def_readwrite("data_type_", &TensorDesc::data_type_)
      .def_readwrite("data_format_", &TensorDesc::data_format_)
      .def_readwrite("shape_", &TensorDesc::shape_)
      .def_readwrite("stride_", &TensorDesc::stride_)
      .def("__str__", [](const TensorDesc &self) {
        std::ostringstream os;
        os << "<nndeploy._nndeploy_internal.device.TensorDesc object at "
           << static_cast<const void *>(&self) << "> : ";
        self.print(os);
        return os.str();
      });
}

}  // namespace device
}  // namespace nndeploy
