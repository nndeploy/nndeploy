
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
      .def("get_size", &BufferDesc::getSize)
      .def("get_size_vector", &BufferDesc::getSizeVector)
      .def("get_real_size", &BufferDesc::getRealSize)
      .def("get_real_size_vector", &BufferDesc::getRealSizeVector)
      .def("get_config", &BufferDesc::getConfig)
      .def("is_same_config", &BufferDesc::isSameConfig)
      .def("is_same_dim", &BufferDesc::isSameDim)
      .def("is_1d", &BufferDesc::is1D)
      .def("print", &BufferDesc::print)
      .def("just_modify",
           py::overload_cast<const size_t &>(&BufferDesc::justModify))
      .def("just_modify",
           py::overload_cast<const base::SizeVector &>(&BufferDesc::justModify))
      .def("just_modify",
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
      // .def("serialize",
      //      [](TensorDesc &self, std::ostream &os) { self.serialize(os); })
      // .def("deserialize",
      //      [](TensorDesc &self, std::istream &is) { self.deserialize(is); })
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
