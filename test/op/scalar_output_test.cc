#include <gtest/gtest.h>

#include "nndeploy/op/op_cast.h"
#include "nndeploy/op/op_gather.h"
#include "nndeploy/op/op_reshape.h"

namespace nndeploy {
TEST(ScalarOutput, ReusesStridedOutputForGatherAndReshape) {
  auto *host = device::getDefaultHostDevice();
  device::TensorDesc vector(base::dataTypeOf<float>(), base::kDataFormatN, {1});
  device::TensorDesc scalar_index(base::dataTypeOf<int64_t>(), base::kDataFormatAuto, {});
  device::TensorDesc empty_shape(base::dataTypeOf<int64_t>(), base::kDataFormatN, {0});
  float value = 7;
  int64_t zero = 0;
  device::Tensor input(host, vector, &value, "input");
  device::Tensor index(host, scalar_index, &zero, "index");
  device::Tensor shape(host, empty_shape, &zero, "shape");
  for (bool gather : {true, false}) {
    device::Tensor output(host, vector);
    auto *data = output.getData();
    ASSERT_FALSE(output.getStride().empty());
    auto status = gather ? op::gather(&input, &index, std::make_shared<ir::GatherParam>(), &output)
                         : op::reshape(&input, &shape, std::make_shared<ir::ReshapeParam>(), &output);
    ASSERT_TRUE(status);
    EXPECT_TRUE(output.getShape().empty());
    EXPECT_TRUE(output.getStride().empty());
    EXPECT_EQ(output.getData(), data);
    EXPECT_EQ(output.getSize(), sizeof(float));
    EXPECT_FLOAT_EQ(*static_cast<float *>(output.getData()), value);
  }
}

TEST(ScalarOutput, GatherExpandsPreviouslyShrunkBuffer) {
  auto *host = device::getDefaultHostDevice();
  device::TensorDesc vector(base::dataTypeOf<float>(), base::kDataFormatN, {2});
  float values[] = {3, 5};
  int64_t ids[] = {1, 0};
  device::Tensor input(host, vector, values, "data"), output(host, vector);
  for (auto dims : {base::IntVector{}, base::IntVector{2}}) {
    device::TensorDesc desc(base::dataTypeOf<int64_t>(), base::kDataFormatAuto, dims);
    device::Tensor index(host, desc, ids, "index");
    ASSERT_TRUE(op::gather(&input, &index, std::make_shared<ir::GatherParam>(), &output));
    EXPECT_EQ(output.getShape(), dims);
  }
  ASSERT_EQ(output.getSize(), sizeof(values));
  EXPECT_FLOAT_EQ(static_cast<float *>(output.getData())[0], 5);
  EXPECT_FLOAT_EQ(static_cast<float *>(output.getData())[1], 3);
}

TEST(CastOutput, RejectsShortBufferWithoutChangingItsDescriptorOrData) {
  auto *host = device::getDefaultHostDevice();
  device::TensorDesc desc(base::dataTypeOf<uint8_t>(), base::kDataFormatN, {4});
  uint8_t data[] = {1, 2, 3, 4};
  device::Tensor input(host, desc, data, "input"), output(host, desc, data, "output");
  auto param = std::make_shared<ir::CastParam>();
  param->to_ = base::dataTypeOf<float>();
  EXPECT_FALSE(op::cast(&input, param, &output));
  EXPECT_EQ(output.getDesc(), desc);
  EXPECT_EQ(data[3], 4);
  device::Tensor valid;
  ASSERT_TRUE(op::cast(&input, param, &valid));
  EXPECT_FLOAT_EQ(static_cast<float *>(valid.getData())[3], 4);
}
}  // namespace nndeploy
