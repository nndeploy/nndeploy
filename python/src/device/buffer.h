#ifndef C5F5CC49_72A4_48A0_B1AB_9AE7116F89AF
#define C5F5CC49_72A4_48A0_B1AB_9AE7116F89AF

#include "device/tensor_util.h"
#include "nndeploy/device/buffer.h"
#include "nndeploy/device/device.h"
#include "nndeploy/device/memory_pool.h"
#include "nndeploy/device/tensor.h"
#include "nndeploy/device/type.h"
#include "nndeploy/device/util.h"
#include "nndeploy_api_registry.h"

namespace nndeploy {
namespace device {

std::string getPyBufferFormat(base::DataType data_type);

py::buffer_info bufferToBufferInfo(device::Buffer *buffer, const py::dtype &dt);

Buffer bufferFromNumpy(const py::array &array);

}  // namespace device
}  // namespace nndeploy

#endif /* C5F5CC49_72A4_48A0_B1AB_9AE7116F89AF */