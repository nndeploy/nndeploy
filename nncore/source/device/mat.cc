
#include "nncore/include/base/status.h"
#include "nncore/include/base/type.h"
#include "nncore/include/device/buffer.h"
#include "nncore/include/device/mat.h"
#include "nncore/include/device/device.h"
#include "nncore/include/device/memory_pool.h"

namespace nncore {
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
}  // namespace nncore
