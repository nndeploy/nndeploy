
#include "nndeploy/op/op_binary.h"

#include "nndeploy/base/any.h"
#include "nndeploy/base/common.h"
#include "nndeploy/base/glic_stl_include.h"
#include "nndeploy/base/log.h"
#include "nndeploy/base/macro.h"
#include "nndeploy/base/object.h"
#include "nndeploy/base/param.h"
#include "nndeploy/base/status.h"
#include "nndeploy/base/string.h"
#include "nndeploy/base/time_profiler.h"
#include "nndeploy/device/buffer.h"
#include "nndeploy/device/device.h"
#include "nndeploy/device/memory_pool.h"
#include "nndeploy/device/tensor.h"
#include "nndeploy/ir/ir.h"
#include "nndeploy/op/op.h"

namespace nndeploy {
namespace op {

base::Status OpBinary::inferShape() {
  base::Status status = base::kStatusCodeOk; // 初始化状态为成功

  // 获取输入张量的形状
  auto input0_shape = inputs_[0]->getShape(); // 输入0的形状
  auto input1_shape = inputs_[1]->getShape(); // 输入1的形状

  // 定义输出形状的变量
  base::IntVector output_shape;

  // 获取两个输入形状的维度大小
  int input0_size = input0_shape.size(); // 输入0的维度数量
  int input1_size = input1_shape.size(); // 输入1的维度数量

  if (input0_size == input1_size) {
    // 如果两个输入的维度数量相同
    for (int i = 0; i < input0_size; i++) {
      if (input0_shape[i] != input1_shape[i]) {
        // 如果当前维度不相等
        if (input0_shape[i] == 1 || input1_shape[i] == 1) {
          // 如果其中一个维度为1，则按照广播规则取最大值
          output_shape.push_back(std::max(input0_shape[i], input1_shape[i]));
        } else {
          // 如果两个维度都不为1且不相等，则无法广播
          NNDEPLOY_LOGE("broadcast failed: dimension mismatch at axis %d.\n", i);
          return base::kStatusCodeErrorInvalidParam;
        }
      } else {
        // 如果当前维度相等，则直接取值
        output_shape.push_back(input0_shape[i]);
      }
    }
  } else {
    // 如果两个输入的维度数量不同
    int max_size = std::max(input0_size, input1_size); // 取较大维度数量
    const base::IntVector &larger_shape =
        (input0_size > input1_size) ? input0_shape : input1_shape; // 较大维度的形状
    const base::IntVector &smaller_shape =
        (input0_size > input1_size) ? input1_shape : input0_shape; // 较小维度的形状

    output_shape.resize(max_size); // 输出形状调整为较大维度的数量

    // 从右向左填充较小的形状
    int diff = max_size - smaller_shape.size(); // 维度差值
    for (int i = max_size - 1; i >= 0; i--) {
      if (i >= diff) { // 对于较小形状有对应维度的部分
        int smaller_idx = i - diff; // 计算较小形状的索引
        if (larger_shape[i] != smaller_shape[smaller_idx]) {
          if (larger_shape[i] == 1 || smaller_shape[smaller_idx] == 1) {
            // 如果其中一个维度为1，则按照广播规则取最大值
            output_shape[i] =
                std::max(larger_shape[i], smaller_shape[smaller_idx]);
          } else {
            // 如果两个维度都不为1且不相等，则无法广播
            NNDEPLOY_LOGE("broadcast failed: dimension mismatch at axis %d.\n", i);
            return base::kStatusCodeErrorInvalidParam;
          }
        } else {
          // 如果当前维度相等，则直接取值
          output_shape[i] = larger_shape[i];
        }
      } else {
        // 对于较小形状没有对应维度的部分，直接取较大形状的值
        output_shape[i] = larger_shape[i];
      }
    }
  }

  // 设置输出张量的形状
  outputs_[0]->reshape(output_shape);

  return status; // 返回状态
}

base::Status OpBinary::inferDataFormat() {
  auto data_format_0 = inputs_[0]->getDataFormat();
  auto data_format_1 = inputs_[1]->getDataFormat();
  auto data_format_output = data_format_0;
  if (data_format_0 != data_format_1) {
    if (data_format_0 > data_format_1) {
      data_format_output = data_format_1;
    } else {
      data_format_output = data_format_0;
    }
  }
  outputs_[0]->setDataFormat(data_format_output);
  return base::kStatusCodeOk;
}

}  // namespace op
}  // namespace nndeploy
