#include <gtest/gtest.h>
#include "nndeploy/depth/drawdepth.h"
#include "nndeploy/depth/depth_anything/depth_anything.h"

namespace nndeploy {
TEST(DepthPostProcess, AcceptsOnePlaneAndRejectsAmbiguousResults) {
  float values[] = {4, -1, 6, 8};
  for (auto shape : {base::IntVector{1, 1, 1, 2}, {2, 1, 1, 2}, {1, 2, 1, 2}}) {
    device::TensorDesc desc(base::dataTypeOf<float>(), base::kDataFormatNCHW, shape);
    device::Tensor tensor(device::getDefaultHostDevice(), desc, values, "depth");
    dag::Edge input("depth"), output("result");
    depth::DepthPostProcess post("post", {&input}, {&output});
    ASSERT_TRUE(input.set(&tensor, true));
    if (shape[0] != 1 || shape[1] != 1) {
      EXPECT_FALSE(post.run());
      continue;
    }
    ASSERT_TRUE(post.run());
    auto *result = dynamic_cast<depth::DepthResult*>(output.getGraphOutputParam());
    ASSERT_NE(result, nullptr);
    EXPECT_EQ(result->data_, (std::vector<float>{4, -1}));
    EXPECT_FLOAT_EQ(result->min_val_, -1);
    EXPECT_FLOAT_EQ(result->max_val_, 4);
  }
}

TEST(DrawDepth, RejectsShortPlaneBeforeRendering) {
  dag::Edge image("image"), depth_edge("depth"), output("output");
  depth::DrawDepth draw("draw", {&image, &depth_edge}, {&output});
  cv::Mat frame(2, 2, CV_8UC3);
  depth::DepthResult result;
  result.width_ = result.height_ = 2;
  result.data_ = {0};
  ASSERT_TRUE(image.set(&frame, true));
  ASSERT_TRUE(depth_edge.set(&result, true));
  EXPECT_FALSE(draw.run());
  result.data_.resize(4);
  EXPECT_TRUE(draw.run());
}
}  // namespace nndeploy
