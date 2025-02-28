
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
    PYBIND11_OVERRIDE_PURE_NAME(base::Status, Architecture, "check_device",
                                checkDevice, device_id, library_path);
  }

  base::Status enableDevice(int device_id = 0,
                            std::string library_path = "") override {
    PYBIND11_OVERRIDE_PURE_NAME(base::Status, Architecture, "enable_device",
                                enableDevice, device_id, library_path);
  }

  base::Status disableDevice() override {
    PYBIND11_OVERRIDE_NAME(base::Status, Architecture, "disable_device",
                           disableDevice);
  }

  Device *getDevice(int device_id) override {
    PYBIND11_OVERRIDE_PURE_NAME(Device *, Architecture, "get_device", getDevice,
                                device_id);
  }

  std::vector<DeviceInfo> getDeviceInfo(
      std::string library_path = "") override {
    PYBIND11_OVERRIDE_PURE_NAME(std::vector<DeviceInfo>, Architecture,
                                "get_device_info", getDeviceInfo, library_path);
  }
};

class PyDevice : public Device {
 public:
  using Device::Device;

  base::DataFormat getDataFormatByShape(const base::IntVector &shape) override {
    PYBIND11_OVERRIDE_NAME(base::DataFormat, Device, "get_data_format_by_shape",
                           getDataFormatByShape, shape);
  }

  BufferDesc toBufferDesc(const TensorDesc &desc,
                          const base::IntVector &config) override {
    PYBIND11_OVERRIDE_PURE_NAME(BufferDesc, Device, "to_buffer_desc",
                                toBufferDesc, desc, config);
  }

  void *allocate(size_t size) override {
    PYBIND11_OVERRIDE_PURE_NAME(void *, Device, "allocate", allocate, size);
  }

  void *allocate(const BufferDesc &desc) override {
    PYBIND11_OVERRIDE_PURE_NAME(void *, Device, "allocate", allocate, desc);
  }

  void deallocate(void *ptr) override {
    PYBIND11_OVERRIDE_PURE_NAME(void, Device, "deallocate", deallocate, ptr);
  }

  void *allocatePinned(size_t size) override {
    PYBIND11_OVERRIDE_NAME(void *, Device, "allocate_pinned", allocatePinned,
                           size);
  }

  void *allocatePinned(const BufferDesc &desc) override {
    PYBIND11_OVERRIDE_NAME(void *, Device, "allocate_pinned", allocatePinned,
                           desc);
  }

  void deallocatePinned(void *ptr) override {
    PYBIND11_OVERRIDE_NAME(void, Device, "deallocate_pinned", deallocatePinned,
                           ptr);
  }

  base::Status copy(void *src, void *dst, size_t size,
                    Stream *stream = nullptr) override {
    PYBIND11_OVERRIDE_PURE_NAME(base::Status, Device, "copy", copy, src, dst,
                                size, stream);
  }

  base::Status download(void *src, void *dst, size_t size,
                        Stream *stream = nullptr) override {
    PYBIND11_OVERRIDE_PURE_NAME(base::Status, Device, "download", download, src,
                                dst, size, stream);
  }

  base::Status upload(void *src, void *dst, size_t size,
                      Stream *stream = nullptr) override {
    PYBIND11_OVERRIDE_PURE_NAME(base::Status, Device, "upload", upload, src,
                                dst, size, stream);
  }

  base::Status copy(Buffer *src, Buffer *dst,
                    Stream *stream = nullptr) override {
    PYBIND11_OVERRIDE_PURE_NAME(base::Status, Device, "copy", copy, src, dst,
                                stream);
  }

  base::Status download(Buffer *src, Buffer *dst,
                        Stream *stream = nullptr) override {
    PYBIND11_OVERRIDE_PURE_NAME(base::Status, Device, "download", download, src,
                                dst, stream);
  }

  base::Status upload(Buffer *src, Buffer *dst,
                      Stream *stream = nullptr) override {
    PYBIND11_OVERRIDE_PURE_NAME(base::Status, Device, "upload", upload, src,
                                dst, stream);
  }

  void *getContext() override {
    PYBIND11_OVERRIDE_NAME(void *, Device, "get_context", getContext);
  }

  Stream *createStream() override {
    PYBIND11_OVERRIDE_NAME(Stream *, Device, "create_stream", createStream);
  }

  Stream *createStream(void *stream) override {
    PYBIND11_OVERRIDE_NAME(Stream *, Device, "create_stream", createStream,
                           stream);
  }

  //   base::Status destroyStream(Stream *stream) override {
  //     PYBIND11_OVERRIDE_NAME(base::Status, Device, "destroy_stream",
  //                            destroyStream, stream);
  //   }

  Event *createEvent() override {
    PYBIND11_OVERRIDE_NAME(Event *, Device, "create_event", createEvent);
  }

  //   base::Status destroyEvent(Event *event) override {
  //     PYBIND11_OVERRIDE_NAME(base::Status, Device, "destroy_event",
  //     destroyEvent,
  //                            event);
  //   }

  //   base::Status createEvents(Event **events, size_t count) override {
  //     PYBIND11_OVERRIDE_NAME(base::Status, Device, "create_events",
  //     createEvents,
  //                            events, count);
  //   }

  //   base::Status destroyEvents(Event **events, size_t count) override {
  //     PYBIND11_OVERRIDE_NAME(base::Status, Device, "destroy_events",
  //                            destroyEvents, events, count);
  //   }

  base::Status init() override {
    PYBIND11_OVERRIDE_PURE_NAME(base::Status, Device, "init", init);
  }

  base::Status deinit() override {
    PYBIND11_OVERRIDE_PURE_NAME(base::Status, Device, "deinit", deinit);
  }
};

NNDEPLOY_API_PYBIND11_MODULE("device", m) {
  // nndeploy::DeviceInfo export as device.DeviceInfo
  py::class_<DeviceInfo>(m, "DeviceInfo", py::dynamic_attr())
      .def(py::init<>())
      .def_readwrite("device_type_", &DeviceInfo::device_type_)
      .def_readwrite("is_support_fp16_", &DeviceInfo::is_support_fp16_);

  // nndeploy::Architecture export as device.Architecture
  py::class_<Architecture, PyArchitecture, std::shared_ptr<Architecture>>(
      m, "Architecture")
      .def(py::init<base::DeviceTypeCode>())
      .def("check_device", &Architecture::checkDevice, py::arg("device_id") = 0,
           py::arg("library_path") = "")
      .def("enable_device", &Architecture::enableDevice,
           py::arg("device_id") = 0, py::arg("library_path") = "")
      .def("disable_device", &Architecture::disableDevice)
      .def("get_device", &Architecture::getDevice, py::arg("device_id"),
           py::return_value_policy::reference)
      .def("get_device_info", &Architecture::getDeviceInfo,
           py::arg("library_path") = "")
      .def("get_device_type_code", &Architecture::getDeviceTypeCode)
      .def("insert_device", &Architecture::insertDevice, py::arg("device_id"),
           py::arg("device"))
      .def("__str__", [](const Architecture &self) {
        std::ostringstream os;
        os << "<nndeploy._nndeploy_internal.device.Architecture object at "
           << static_cast<const void *>(&self)
           << "> : " << base::deviceTypeCodeToString(self.getDeviceTypeCode());
        return os.str();
      });

  // nndeploy::Device export as device.Device
  py::class_<Device, PyDevice>(m, "Device")
      .def(py::init<base::DeviceType, std::string>(), py::arg("device_type"),
           py::arg("library_path") = "")
      .def("get_data_format_by_shape", &Device::getDataFormatByShape,
           py::arg("shape"))
      .def("to_buffer_desc", &Device::toBufferDesc, py::arg("desc"),
           py::arg("config"))
      // py::return_value_policy::reference, must be call by deallocate
      .def("allocate", py::overload_cast<size_t>(&Device::allocate),
           py::arg("size"), py::return_value_policy::reference)
      .def("allocate", py::overload_cast<const BufferDesc &>(&Device::allocate),
           py::arg("desc"), py::return_value_policy::reference)
      .def("deallocate", &Device::deallocate, py::arg("ptr"))
      .def("allocate_pinned",
           py::overload_cast<size_t>(&Device::allocatePinned), py::arg("size"),
           py::return_value_policy::reference)
      .def("allocate_pinned",
           py::overload_cast<const BufferDesc &>(&Device::allocatePinned),
           py::arg("desc"), py::return_value_policy::reference)
      .def("deallocate_pinned", &Device::deallocatePinned, py::arg("ptr"))
      .def("copy",
           py::overload_cast<void *, void *, size_t, Stream *>(&Device::copy),
           py::arg("src"), py::arg("dst"), py::arg("size"),
           py::arg("stream") = nullptr)
      .def("download",
           py::overload_cast<void *, void *, size_t, Stream *>(
               &Device::download),
           py::arg("src"), py::arg("dst"), py::arg("size"),
           py::arg("stream") = nullptr)
      .def("upload",
           py::overload_cast<void *, void *, size_t, Stream *>(&Device::upload),
           py::arg("src"), py::arg("dst"), py::arg("size"),
           py::arg("stream") = nullptr)
      .def("copy",
           py::overload_cast<Buffer *, Buffer *, Stream *>(&Device::copy),
           py::arg("src"), py::arg("dst"), py::arg("stream") = nullptr)
      .def("download",
           py::overload_cast<Buffer *, Buffer *, Stream *>(&Device::download),
           py::arg("src"), py::arg("dst"), py::arg("stream") = nullptr)
      .def("upload",
           py::overload_cast<Buffer *, Buffer *, Stream *>(&Device::upload),
           py::arg("src"), py::arg("dst"), py::arg("stream") = nullptr)
      .def("get_context", &Device::getContext,
           py::return_value_policy::reference)
      .def("create_stream", py::overload_cast<>(&Device::createStream),
           py::return_value_policy::take_ownership)
      .def("create_stream", py::overload_cast<void *>(&Device::createStream),
           py::arg("stream"), py::return_value_policy::take_ownership)
      // .def("destroy_stream", &Device::destroyStream,
      // py::arg("stream"))
      .def("create_event", &Device::createEvent,
           py::return_value_policy::take_ownership)
      // .def("destroy_event", &Device::destroyEvent, py::arg("event"))
      .def(
          "create_events",
          [](Device &self, std::vector<Event *> &events) {
            for (size_t i = 0; i < events.size(); i++) {
              events[i] = self.createEvent();
            }
            return base::kStatusCodeOk;
          },
          py::arg("events"))
      // .def(
      //     "destroy_events",
      //     [](Device &self, std::vector<Event *> &events) {
      //       return self.destroyEvents(events.data(), events.size());
      //     },
      //     py::arg("events"))
      .def("get_device_type", &Device::getDeviceType)
      .def("init", &Device::init)
      .def("deinit", &Device::deinit)
      .def("__str__", [](const Device &self) {
        std::ostringstream os;
        os << "<nndeploy._nndeploy_internal.device.Device object at "
           << static_cast<const void *>(&self)
           << "> : " << base::deviceTypeToString(self.getDeviceType());
        return os.str();
      });

  py::class_<Stream>(m, "Stream")
      .def(py::init<Device *>())
      .def(py::init<Device *, void *>())
      .def("get_device_type", &Stream::getDeviceType)
      .def("get_device", &Stream::getDevice, py::return_value_policy::reference)
      .def("synchronize", &Stream::synchronize)
      .def("record_event", &Stream::recordEvent)
      .def("wait_event", &Stream::waitEvent)
      .def("on_execution_context_setup", &Stream::onExecutionContextSetup)
      .def("on_execution_context_teardown", &Stream::onExecutionContextTeardown)
      .def("get_native_stream", &Stream::getNativeStream,
           py::return_value_policy::reference)
      .def("__str__", [](const Stream &self) {
        std::ostringstream os;
        os << "<nndeploy._nndeploy_internal.device.Stream object at "
           << static_cast<const void *>(&self)
           << "> : " << base::deviceTypeToString(self.getDeviceType());
        return os.str();
      });

  py::class_<Event>(m, "Event")
      .def(py::init<Device *>())
      .def("get_device_type", &Event::getDeviceType)
      .def("get_device", &Event::getDevice, py::return_value_policy::reference)
      .def("query_done", &Event::queryDone)
      .def("synchronize", &Event::synchronize)
      .def("get_native_event", &Event::getNativeEvent,
           py::return_value_policy::reference)
      .def("__str__", [](const Event &self) {
        std::ostringstream os;
        os << "<nndeploy._nndeploy_internal.device.Stream object at "
           << static_cast<const void *>(&self)
           << "> : " << base::deviceTypeToString(self.getDeviceType());
        return os.str();
      });

  // 定义注册函数
  m.def("register_architecture",
        [](base::DeviceTypeCode device_type_code, py::object py_architecture) {
          // 获取 Architecture 的 shared_ptr
          std::shared_ptr<Architecture> architecture =
              py_architecture.cast<std::shared_ptr<Architecture>>();
          // 注册到 ArchitectureMap
          getArchitectureMap()[device_type_code] = architecture;
        });

  m.def("print_architecture_map", []() {
    for (const auto &item : getArchitectureMap()) {
      std::cout << base::deviceTypeCodeToString(item.first) << " : "
                << (static_cast<Architecture *>(item.second.get()))
                       ->getDeviceTypeCode()
                << std::endl;
    }
  });

  // export as device.get_architecture
  m.def("get_architecture", &getArchitectureSharedPtr,
        "Get the Architecture of the specified type",
        py::arg("device_type_code"));

  // export as device.get_default_host_device
  m.def("get_default_host_device", &getDefaultHostDevice,
        py::return_value_policy::reference, "Get the default host device");

  // export as device.is_host_device_type
  m.def("is_host_device_type", &isHostDeviceType,
        "Check if a device type is a host device type", py::arg("device_type"));

  // export as device.check_device
  m.def("check_device", &checkDevice, "Check if a device is available",
        py::arg("device_type"), py::arg("library_path") = "");

  // export as device.enable_device
  m.def("enable_device", &enableDevice, "Enable a device",
        py::arg("device_type"), py::arg("library_path") = "");

  // export as device.get_device
  m.def("get_device", &getDevice, "A function which gets a device by type",
        py::arg("device_type"), py::return_value_policy::reference,
        "Get a device by type");

  // export as device.create_stream
  m.def("create_stream", py::overload_cast<base::DeviceType>(&createStream),
        "Create a stream for a device type", py::arg("device_type"),
        py::return_value_policy::take_ownership);
  m.def("create_stream",
        py::overload_cast<base::DeviceType, void *>(&createStream),
        "Create a stream for a device type with an existing stream",
        py::arg("device_type"), py::arg("stream"),
        py::return_value_policy::take_ownership);

  // export as device.destroy_stream
  // m.def("destroy_stream", &destroyStream, "Destroy a stream",
  // py::arg("stream"));

  // export as device.create_event
  m.def("create_event", &createEvent, "Create an event for a device type",
        py::arg("device_type"), py::return_value_policy::take_ownership);

  // export as device.destroy_event
  // m.def("destroy_event", &destroyEvent, "Destroy an event",
  // py::arg("event"));

  // export as device.create_events
  m.def(
      "create_events",
      [](base::DeviceType device_type, std::vector<Event *> &events) {
        return createEvents(device_type, events.data(), events.size());
      },
      py::arg("device_type"), py::arg("events"));

  // export as device.destroy_events
  // m.def("destroy_events", &destroyEvents,
  //       "Destroy multiple events for a device type", py::arg("device_type"),
  //       py::arg("events"), py::arg("count"));

  // export as device.get_device_info
  m.def("get_device_info", &getDeviceInfo,
        "Get device info for a device type code", py::arg("device_type_code"),
        py::arg("library_path") = "");

  // export as device.disable_device
  m.def("disable_device", &disableDevice, "Disable current device");

  // export as device.destory_architecture
  m.def("destory_architecture", &destoryArchitecture,
        "Destroy device architecture");
}

}  // namespace device
}  // namespace nndeploy