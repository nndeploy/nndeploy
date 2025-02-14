
#include "nndeploy/device/device.h"

#include "nndeploy/device/buffer.h"
#include "nndeploy/device/memory_pool.h"
#include "nndeploy/device/tensor.h"
#include "nndeploy/device/type.h"
#include "nndeploy/device/util.h"
#include "nndeploy_api_registry.h"

namespace nndeploy {
namespace device {

class PyArchitecture : public Architecture {
 public:
  using Architecture::Architecture;

  base::Status checkDevice(int device_id = 0,
                           std::string library_path = "") override {
    PYBIND11_OVERRIDE_PURE(base::Status, Architecture, checkDevice, device_id,
                           library_path);
  }

  base::Status enableDevice(int device_id = 0,
                            std::string library_path = "") override {
    PYBIND11_OVERRIDE_PURE(base::Status, Architecture, enableDevice, device_id,
                           library_path);
  }

  Device *getDevice(int device_id) override {
    PYBIND11_OVERRIDE_PURE(Device *, Architecture, getDevice, device_id);
  }

  std::vector<DeviceInfo> getDeviceInfo(
      std::string library_path = "") override {
    PYBIND11_OVERRIDE_PURE(std::vector<DeviceInfo>, Architecture, getDeviceInfo,
                           library_path);
  }
};

class PyDevice : public Device {
 public:
  using Device::Device;

  BufferDesc toBufferDesc(const TensorDesc &desc,
                          const base::IntVector &config) override {
    PYBIND11_OVERRIDE_PURE(BufferDesc, Device, toBufferDesc, desc, config);
  }

  void *allocate(size_t size) override {
    PYBIND11_OVERRIDE_PURE(void *, Device, allocate, size);
  }

  void *allocate(const BufferDesc &desc) override {
    PYBIND11_OVERRIDE_PURE(void *, Device, allocate, desc);
  }

  void deallocate(void *ptr) override {
    PYBIND11_OVERRIDE_PURE(void, Device, deallocate, ptr);
  }

  base::Status copy(void *src, void *dst, size_t size,
                    Stream *stream = nullptr) override {
    PYBIND11_OVERRIDE_PURE(base::Status, Device, copy, src, dst, size, stream);
  }

  base::Status download(void *src, void *dst, size_t size,
                        Stream *stream = nullptr) override {
    PYBIND11_OVERRIDE_PURE(base::Status, Device, download, src, dst, size,
                           stream);
  }

  base::Status upload(void *src, void *dst, size_t size,
                      Stream *stream = nullptr) override {
    PYBIND11_OVERRIDE_PURE(base::Status, Device, upload, src, dst, size,
                           stream);
  }

  base::Status copy(Buffer *src, Buffer *dst,
                    Stream *stream = nullptr) override {
    PYBIND11_OVERRIDE_PURE(base::Status, Device, copy, src, dst, stream);
  }

  base::Status download(Buffer *src, Buffer *dst,
                        Stream *stream = nullptr) override {
    PYBIND11_OVERRIDE_PURE(base::Status, Device, download, src, dst, stream);
  }

  base::Status upload(Buffer *src, Buffer *dst,
                      Stream *stream = nullptr) override {
    PYBIND11_OVERRIDE_PURE(base::Status, Device, upload, src, dst, stream);
  }

  base::Status init() override {
    PYBIND11_OVERRIDE_PURE(base::Status, Device, init);
  }

  base::Status deinit() override {
    PYBIND11_OVERRIDE_PURE(base::Status, Device, deinit);
  }
};

NNDEPLOY_API_PYBIND11_MODULE("device", m) {
  // nndeploy::device::DeviceInfo export as device.DeviceInfo
  py::class_<device::DeviceInfo>(m, "DeviceInfo", py::dynamic_attr())
      .def(py::init<>())
      .def_readwrite("device_type_", &device::DeviceInfo::device_type_)
      .def_readwrite("is_support_fp16_", &device::DeviceInfo::is_support_fp16_);

  // nndeploy::device::Architecture export as device.Architecture
  py::class_<device::Architecture, PyArchitecture,
             std::shared_ptr<device::Architecture>>(m, "Architecture")
      .def(py::init<base::DeviceTypeCode>())
      .def("checkDevice", &device::Architecture::checkDevice,
           py::arg("device_id") = 0, py::arg("library_path") = "")
      .def("enableDevice", &device::Architecture::enableDevice,
           py::arg("device_id") = 0, py::arg("library_path") = "")
      .def("disableDevice", &device::Architecture::disableDevice)
      .def("getDevice", &device::Architecture::getDevice, py::arg("device_id"),
           py::return_value_policy::reference)
      .def("getDeviceInfo", &device::Architecture::getDeviceInfo,
           py::arg("library_path") = "")
      .def("getDeviceTypeCode", &device::Architecture::getDeviceTypeCode)
      .def("__str__", [](const Architecture &self) {
        std::ostringstream os;
        os << "<nndeploy._nndeploy_internal.device.Architecture object at "
           << static_cast<const void *>(&self)
           << "> : " << base::deviceTypeCodeToString(self.getDeviceTypeCode());
        return os.str();
      });

  // nndeploy::device::Device export as device.Device
  py::class_<device::Device, PyDevice, std::shared_ptr<device::Device>>(
      m, "Device", py::dynamic_attr())
      .def(py::init<base::DeviceType, std::string>(), py::arg("device_type"),
           py::arg("library_path") = "")
      .def("getDataFormatByShape", &device::Device::getDataFormatByShape,
           py::arg("shape"))
      .def("toBufferDesc", &device::Device::toBufferDesc, py::arg("desc"),
           py::arg("config"))
      .def("allocate", py::overload_cast<size_t>(&device::Device::allocate),
           py::arg("size"), py::return_value_policy::reference)
      .def("allocate",
           py::overload_cast<const BufferDesc &>(&device::Device::allocate),
           py::arg("desc"), py::return_value_policy::reference)
      .def("deallocate", &device::Device::deallocate, py::arg("ptr"))
      .def("allocatePinned",
           py::overload_cast<size_t>(&device::Device::allocatePinned),
           py::arg("size"), py::return_value_policy::reference)
      .def("allocatePinned",
           py::overload_cast<const BufferDesc &>(
               &device::Device::allocatePinned),
           py::arg("desc"), py::return_value_policy::reference)
      .def("deallocatePinned", &device::Device::deallocatePinned,
           py::arg("ptr"))
      .def("copy",
           py::overload_cast<void *, void *, size_t, Stream *>(
               &device::Device::copy),
           py::arg("src"), py::arg("dst"), py::arg("size"),
           py::arg("stream") = nullptr)
      .def("download",
           py::overload_cast<void *, void *, size_t, Stream *>(
               &device::Device::download),
           py::arg("src"), py::arg("dst"), py::arg("size"),
           py::arg("stream") = nullptr)
      .def("upload",
           py::overload_cast<void *, void *, size_t, Stream *>(
               &device::Device::upload),
           py::arg("src"), py::arg("dst"), py::arg("size"),
           py::arg("stream") = nullptr)
      .def("copy",
           py::overload_cast<Buffer *, Buffer *, Stream *>(
               &device::Device::copy),
           py::arg("src"), py::arg("dst"), py::arg("stream") = nullptr)
      .def("download",
           py::overload_cast<Buffer *, Buffer *, Stream *>(
               &device::Device::download),
           py::arg("src"), py::arg("dst"), py::arg("stream") = nullptr)
      .def("upload",
           py::overload_cast<Buffer *, Buffer *, Stream *>(
               &device::Device::upload),
           py::arg("src"), py::arg("dst"), py::arg("stream") = nullptr)
      .def("getContext", &device::Device::getContext,
           py::return_value_policy::reference)
      .def("createStream", py::overload_cast<>(&device::Device::createStream))
      .def("createStream",
           py::overload_cast<void *>(&device::Device::createStream),
           py::arg("stream"))
      // .def("destroyStream", &device::Device::destroyStream,
      // py::arg("stream"))
      .def("createEvent", &device::Device::createEvent)
      // .def("destroyEvent", &device::Device::destroyEvent, py::arg("event"))
      .def(
          "createEvents",
          [](device::Device &self, std::vector<Event *> &events) {
            return self.createEvents(events.data(), events.size());
          },
          py::arg("events"))
      // .def(
      //     "destroyEvents",
      //     [](device::Device &self, std::vector<Event *> &events) {
      //       return self.destroyEvents(events.data(), events.size());
      //     },
      //     py::arg("events"))
      .def("getDeviceType", &device::Device::getDeviceType)
      .def("init", &device::Device::init)
      .def("deinit", &device::Device::deinit);

  py::class_<device::Stream>(m, "Stream", py::dynamic_attr())
      .def(py::init<device::Device *>())
      .def(py::init<device::Device *, void *>())
      .def("getDeviceType", &device::Stream::getDeviceType)
      .def("getDevice", &device::Stream::getDevice,
           py::return_value_policy::reference)
      .def("synchronize", &device::Stream::synchronize)
      .def("recordEvent", &device::Stream::recordEvent)
      .def("waitEvent", &device::Stream::waitEvent)
      .def("onExecutionContextSetup", &device::Stream::onExecutionContextSetup)
      .def("onExecutionContextTeardown",
           &device::Stream::onExecutionContextTeardown)
      .def("getCommandQueue", &device::Stream::getCommandQueue,
           py::return_value_policy::reference);

  py::class_<device::Event>(m, "Event")
      .def(py::init<device::Device *>())
      .def("getDeviceType", &device::Event::getDeviceType)
      .def("getDevice", &device::Event::getDevice,
           py::return_value_policy::reference)
      .def("queryDone", &device::Event::queryDone)
      .def("synchronize", &device::Event::synchronize);

  // 定义注册函数
  m.def("registerArchitecture",
        [](base::DeviceTypeCode device_type_code, py::object py_architecture) {
          // 获取 Architecture 的 shared_ptr
          std::shared_ptr<device::Architecture> architecture =
              py_architecture.cast<std::shared_ptr<device::Architecture>>();
          // 注册到 ArchitectureMap
          getArchitectureMap()[device_type_code] = architecture;
        });

  m.def("printArchitectureMap", []() {
    for (const auto &item : getArchitectureMap()) {
      std::cout << base::deviceTypeCodeToString(item.first) << " : "
                << (static_cast<Architecture *>(item.second.get()))
                       ->getDeviceTypeCode()
                << std::endl;
    }
  });

  // export as device.getArchitecture
  m.def("getArchitecture", &getArchitecture, py::return_value_policy::reference,
        "Get the Architecture of the specified type",
        py::arg("device_type_code"));

  // export as device.getDefaultHostDevice
  m.def("getDefaultHostDevice", &getDefaultHostDevice,
        py::return_value_policy::reference, "Get the default host device");

  // export as device.isHostDeviceType
  m.def("isHostDeviceType", &isHostDeviceType,
        "Check if a device type is a host device type", py::arg("device_type"));

  // export as device.checkDevice
  m.def("checkDevice", &checkDevice, "Check if a device is available",
        py::arg("device_type"), py::arg("library_path") = "");

  // export as device.enableDevice
  m.def("enableDevice", &enableDevice, "Enable a device",
        py::arg("device_type"), py::arg("library_path") = "");

  // export as device.getDevice
  m.def("getDevice", &getDevice, "A function which gets a device by type",
        py::arg("device_type"), py::return_value_policy::reference,
        "Get a device by type");

  // export as device.createStream
  m.def("createStream", py::overload_cast<base::DeviceType>(&createStream),
        "Create a stream for a device type", py::arg("device_type"));
  m.def("createStream",
        py::overload_cast<base::DeviceType, void *>(&createStream),
        "Create a stream for a device type with an existing stream",
        py::arg("device_type"), py::arg("stream"));

  // export as device.destroyStream
  // m.def("destroyStream", &destroyStream, "Destroy a stream",
  // py::arg("stream"));

  // export as device.createEvent
  m.def("createEvent", &createEvent, "Create an event for a device type",
        py::arg("device_type"));

  // export as device.destroyEvent
  // m.def("destroyEvent", &destroyEvent, "Destroy an event", py::arg("event"));

  // export as device.createEvents
  m.def(
      "createEvents",
      [](base::DeviceType device_type, std::vector<Event *> &events) {
        return createEvents(device_type, events.data(), events.size());
      },
      py::arg("device_type"), py::arg("events"));

  // export as device.destroyEvents
  // m.def("destroyEvents", &destroyEvents,
  //       "Destroy multiple events for a device type", py::arg("device_type"),
  //       py::arg("events"), py::arg("count"));

  // export as device.getDeviceInfo
  m.def("getDeviceInfo", &getDeviceInfo,
        "Get device info for a device type code", py::arg("device_type_code"),
        py::arg("library_path") = "");

  // // export as device.disableDevice
  // m.def("disableDevice", &disableDevice, "Disable current device");

  // // export as device.destoryArchitecture
  // m.def("destoryArchitecture", &destoryArchitecture,
  //       "Destroy device architecture");
}

}  // namespace device
}  // namespace nndeploy