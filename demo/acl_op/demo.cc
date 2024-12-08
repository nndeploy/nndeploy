#include "acl/acl.h"
extern void hello_world_do(uint32_t coreDim, void* stream);

int32_t main(int argc, char const* argv[]) {
  // AscendCL初始化
  aclInit(nullptr);
  // 运行管理资源申请
  int32_t deviceId = 0;
  aclrtSetDevice(deviceId);
  aclrtStream stream = nullptr;
  aclrtCreateStream(&stream);

  // 设置参与运算的核数为8
  constexpr uint32_t blockDim = 8;
  // 用内核调用符<<<>>>调用核函数，hello_world_do中封装了<<<>>>调用
  hello_world_do(blockDim, stream);
  aclrtSynchronizeStream(stream);
  // 资源释放和AscendCL去初始化
  aclrtDestroyStream(stream);
  aclrtResetDevice(deviceId);
  aclFinalize();
  return 0;
}