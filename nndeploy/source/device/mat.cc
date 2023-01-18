
#include "nndeploy/include/base/status.h"
#include "nndeploy/include/base/type.h"
#include "nndeploy/include/device/buffer.h"
#include "nndeploy/include/device/mat.h"
#include "nndeploy/include/device/device.h"
#include "nndeploy/include/device/memory_pool.h"

namespace nndeploy {
namespace device {


class Mat {
 public:
  Mat();
  virtual ~Mat();

  // get
  bool empty();
  base::DeviceType getDeviceType();
  int32_t getId();
  void *getPtr();

 private:
  MatDesc desc_;

  Buffer *buffer_;

  // 引用计数 + 满足多线程
};

}  // namespace device
}  // namespace nndeploy
