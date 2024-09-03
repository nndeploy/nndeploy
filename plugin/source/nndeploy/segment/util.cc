#include "nndeploy/model/segment/util.h"

namespace nndeploy {
namespace model {

/// @brief 将一个float vector转换为当前设备的tensor
/// @param data 代转换数组
/// @param dst_shape tensor形状
/// @param device 设备指针
/// @param data_format 数据排布格式
/// @param name tensor名称
/// @return
NNDEPLOY_CC_API device::Tensor *convertVectorToTensor(
    std::vector<float> &data, std::vector<int> dst_shape,
    device::Device *device, base::DataFormat data_format, std::string name) {
  base::IntVector shape = dst_shape;

  base::DataType data_type = base::dataTypeOf<float>();
  base::SizeVector stride = base::SizeVector();
  device::TensorDesc tensor_desc(data_type, data_format, dst_shape, stride);

  device::Tensor *tensor =
      new device::Tensor(device, tensor_desc, data.data(), name);

  return tensor;
}
}  // namespace model
}  // namespace nndeploy