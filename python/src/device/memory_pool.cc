
#include "nndeploy/device/memory_pool.h"

#include "nndeploy/device/buffer.h"
#include "nndeploy/device/device.h"
#include "nndeploy/device/tensor.h"
#include "nndeploy/device/type.h"
#include "nndeploy/device/util.h"
#include "nndeploy_api_registry.h"

namespace nndeploy {
namespace device {

class PyMemoryPool : public MemoryPool {
 public:
  using MemoryPool::MemoryPool;

  base::Status deinit() override {
    PYBIND11_OVERRIDE_PURE(base::Status, MemoryPool, deinit);
  }

  void *allocate(size_t size) override {
    PYBIND11_OVERRIDE_PURE(void *, MemoryPool, allocate, size);
  }

  void *allocate(const BufferDesc &desc) override {
    PYBIND11_OVERRIDE_PURE(void *, MemoryPool, allocate, desc);
  }

  void deallocate(void *ptr) override {
    PYBIND11_OVERRIDE_PURE(void, MemoryPool, deallocate, ptr);
  }

  void *allocatePinned(size_t size) override {
    PYBIND11_OVERRIDE_PURE(void *, MemoryPool, allocatePinned, size);
  }

  void *allocatePinned(const BufferDesc &desc) override {
    PYBIND11_OVERRIDE_PURE(void *, MemoryPool, allocatePinned, desc);
  }

  void deallocatePinned(void *ptr) override {
    PYBIND11_OVERRIDE_PURE(void, MemoryPool, deallocatePinned, ptr);
  }
};

NNDEPLOY_API_PYBIND11_MODULE("device", m) {
  py::class_<MemoryPool, PyMemoryPool, std::shared_ptr<MemoryPool>>(
      m, "MemoryPool", py::dynamic_attr())
      .def(py::init<Device *, base::MemoryPoolType>(), py::arg("device"),
           py::arg("memory_pool_type"),
           "Constructor, create a MemoryPool object")
      .def("init", py::overload_cast<>(&MemoryPool::init),
           "Initialize the MemoryPool")
      .def("init", py::overload_cast<size_t>(&MemoryPool::init),
           py::arg("size"), "Initialize the MemoryPool with specified size")
      .def("init", py::overload_cast<void *, size_t>(&MemoryPool::init),
           py::arg("ptr"), py::arg("size"),
           "Initialize the MemoryPool with specified memory pointer and size")
      .def("init", py::overload_cast<Buffer *>(&MemoryPool::init),
           py::arg("buffer"),
           "Initialize the MemoryPool with specified Buffer object")
      .def("deinit", &MemoryPool::deinit, "Deinitialize the MemoryPool")
      .def("allocate", py::overload_cast<size_t>(&MemoryPool::allocate),
           py::return_value_policy::reference, py::arg("size"),
           "Allocate memory with specified size from the MemoryPool")
      .def("allocate",
           py::overload_cast<const BufferDesc &>(&MemoryPool::allocate),
           py::return_value_policy::reference, py::arg("desc"),
           "Allocate memory with specified BufferDesc from the MemoryPool")
      .def("deallocate", &MemoryPool::deallocate, py::arg("ptr"),
           "Deallocate memory allocated from the MemoryPool")
      .def("allocatePinned",
           py::overload_cast<size_t>(&MemoryPool::allocatePinned),
           py::return_value_policy::reference, py::arg("size"),
           "Allocate pinned memory with specified size from the MemoryPool")
      .def("allocatePinned",
           py::overload_cast<const BufferDesc &>(&MemoryPool::allocatePinned),
           py::return_value_policy::reference, py::arg("desc"),
           "Allocate pinned memory with specified BufferDesc from the "
           "MemoryPool")
      .def("deallocatePinned", &MemoryPool::deallocatePinned, py::arg("ptr"),
           "Deallocate pinned memory allocated from the MemoryPool")
      .def("getDevice", &MemoryPool::getDevice,
           py::return_value_policy::reference,
           "Get the Device object associated with the MemoryPool")
      .def("getMemoryPoolType", &MemoryPool::getMemoryPoolType,
           "Get the type of the MemoryPool");
}

}  // namespace device
}  // namespace nndeploy