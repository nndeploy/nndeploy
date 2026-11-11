#include <gtest/gtest.h>

#include "nndeploy/super_resolution/super_resolution.h"

namespace nndeploy {
TEST(SuperResolution, ConvertsEveryFrameFromPlanarRgbToBgr) {
  float rgb[] = {1, 0, 0, 0, 0, 1};
  device::TensorDesc desc(base::dataTypeOf<float>(), base::kDataFormatNCHW,
                          {2, 3, 1, 1});
  device::Tensor tensor(device::getDefaultHostDevice(), desc, rgb, "rgb");
  dag::Edge input("in"), output("out");
  super_resolution::SuperResolutionPostProcess post("post", {&input}, {&output});
  ASSERT_TRUE(input.set(&tensor, true));
  ASSERT_TRUE(post.run());
  auto* frames = output.getGraphOutput<std::vector<cv::Mat>>();
  ASSERT_NE(frames, nullptr);
  ASSERT_EQ(frames->size(), 2u);
  EXPECT_EQ(frames->at(0).at<cv::Vec3b>(0, 0), cv::Vec3b(0, 0, 255));
  EXPECT_EQ(frames->at(1).at<cv::Vec3b>(0, 0), cv::Vec3b(255, 0, 0));
}
}  // namespace nndeploy
