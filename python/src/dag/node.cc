#include "nndeploy/dag/node.h"

#include "nndeploy/dag/edge.h"
#include "nndeploy_api_registry.h"

namespace py = pybind11;
namespace nndeploy {
namespace dag {

// 创建一个 Node 的 trampoline 类来处理虚函数
class PyNode : public Node {
 public:
  // 继承构造函数
  using Node::Node;

  // 必须实现默认构造函数，因为 pybind11 需要它
  PyNode() : Node("", nullptr, nullptr) {}

  // 重写虚函数，将调用转发到 Python
  base::Status run() override {
    PYBIND11_OVERRIDE_PURE(base::Status,  // 返回值类型
                           Node,          // 父类
                           run            // 函数名
    );
    return base::kStatusCodeOk;  // 永远不会执行到这里，只是为了编译通过
  }
};

NNDEPLOY_API_PYBIND11_MODULE("dag", m) {
  // 定义Node类绑定
  py::class_<Node, PyNode, std::shared_ptr<Node>>(m, "Node", py::dynamic_attr())
      // 构造函数
      .def(py::init<const std::string&>())
      .def(py::init<const std::string&, Edge*, Edge*>())
      .def(py::init<const std::string&, std::initializer_list<Edge*>,
                    std::initializer_list<Edge*>>())
      .def(py::init<const std::string&, std::vector<Edge*>,
                    std::vector<Edge*>>())

      // 基本属性访问
      .def("getName", &Node::getName)
      .def("getDeviceType", &Node::getDeviceType)
      .def("setDeviceType", &Node::setDeviceType)

      // 参数相关
      .def("setParam", &Node::setParam)
      .def("getParam", &Node::getParam)
      .def("setExternalParam", &Node::setExternalParam)

      // 输入输出边
      .def("getInput", &Node::getInput)
      .def("getOutput", &Node::getOutput)
      .def("getAllInput", &Node::getAllInput)
      .def("getAllOutput", &Node::getAllOutput)

      // 状态标志
      .def("getConstructed", &Node::getConstructed)
      .def("setParallelType", &Node::setParallelType)
      .def("getParallelType", &Node::getParallelType)
      .def("setInnerFlag", &Node::setInnerFlag)
      .def("setInitializedFlag", &Node::setInitializedFlag)
      .def("getInitialized", &Node::getInitialized)
      .def("setTimeProfileFlag", &Node::setTimeProfileFlag)
      .def("getTimeProfileFlag", &Node::getTimeProfileFlag)
      .def("setDebugFlag", &Node::setDebugFlag)
      .def("getDebugFlag", &Node::getDebugFlag)
      .def("setRunningFlag", &Node::setRunningFlag)
      .def("isRunning", &Node::isRunning)

      // 核心功能方法
      .def("init", &Node::init)
      .def("deinit", &Node::deinit)
      .def("getMemorySize", &Node::getMemorySize)
      .def("setMemory", &Node::setMemory)
      .def("updataInput", &Node::updataInput)
      .def("run", &Node::run);
}

}  // namespace dag
}  // namespace nndeploy