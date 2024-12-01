#include "nndeploy/net/net.h"

#include <pybind11/iostream.h>
#include <pybind11/stl.h>

#include "nndeploy_api_registry.h"

namespace nndeploy {

namespace net {

NNDEPLOY_API_PYBIND11_MODULE("net", m) {
  py::class_<Net, std::shared_ptr<Net>>(m, "Net")
      .def(py::init<>())
      .def("setModelDesc", &Net::setModelDesc)
      .def("setDeviceType", &Net::setDeviceType)
      .def("init", &Net::init)
      .def(
          "dump",
          [](Net& self, const std::string& file_path) {
            // 打开文件，如果不存在则创建，如果存在则覆盖
            std::ofstream oss(file_path);
            if (!oss.is_open()) {
              throw std::runtime_error("Failed to open file: " + file_path);
            }

            self.dump(oss);
          },
          py::arg("file_path"))
      .def("preRun", &Net::preRun)
      .def("run", &Net::run)
      .def("postRun", &Net::postRun)
      .def("deinit", &Net::deinit)
      .def("getAllOutput", &Net::getAllOutput)
      .def("getAllInput", &Net::getAllInput)
      .def("enableOpt", &Net::enableOpt)
      .def("setEnablePass", &Net::setEnablePass)
      .def("setDisablePass", &Net::setDisablePass)
      .def("setInputs", [](Net& self,
                           const py::dict& inputs_map) {  // 使用深拷贝
        std::vector<device::Tensor*> inputs = self.getAllInput();
        for (device::Tensor* input : inputs) {
          // 获取输入张量的名称
          std::string input_name = input->getName();
          // 检查输入名称是否存在于inputs_map中
          if (inputs_map.contains(input_name)) {
            // 将std::string键转换为py::object
            py::object py_key = py::cast(input_name);
            // 获取与名称对应的Tensor对象
            py::object py_tensor_obj = inputs_map[py_key];
            // 将py::object转换为device::Tensor*类型
            device::Tensor* py_tensor = py_tensor_obj.cast<device::Tensor*>();
            // 执行复制操作
            py_tensor->copyTo(input);
          }
        }
      });
}

}  // namespace net

}  // namespace nndeploy