#include <gtest/gtest.h>
#include "nndeploy/preprocess/batch_preprocess.h"

namespace nndeploy {
namespace preprocess {
TEST(BatchPreprocessTest, CopiesEachSampleAndRejectsMismatchedShapes) {
  dag::Edge input("images"), output("tensor");
  BatchPreprocess node("batch", {&input}, {&output});
  ASSERT_TRUE(node.setNodeKey("nndeploy::preprocess::CvtNormTrans"));
  std::vector<cv::Mat> images{cv::Mat(2, 3, CV_8UC3, cv::Scalar::all(0)),
                              cv::Mat(2, 3, CV_8UC3, cv::Scalar::all(255))};
  ASSERT_TRUE(input.set(&images, true));
  ASSERT_TRUE(node.run());
  auto *tensor = output.getTensor(&node);
  ASSERT_NE(tensor, nullptr);
  EXPECT_EQ(tensor->getShape(), (base::IntVector{2, 3, 2, 3}));
  auto *data = static_cast<float *>(tensor->getData());
  for (int i = 0; i < 36; ++i) EXPECT_FLOAT_EQ(data[i], i < 18 ? 0.f : 1.f);

  images[1] = cv::Mat(3, 3, CV_8UC3, cv::Scalar::all(255));
  EXPECT_FALSE(node.run());
  EXPECT_EQ(output.getTensor(&node), tensor);
}

class StridedSample : public dag::Node {
 public:
  using Node::Node;
  base::Status run() override {
    auto *image = getInput()->getCvMat(this);
    const bool planar = image->rows == 1;
    const auto format = planar ? base::kDataFormatNCHW : base::kDataFormatNHWC;
    device::TensorDesc desc(base::dataTypeOf<uint8_t>(), format,
        planar ? base::IntVector{1, 1, 1, 2} : base::IntVector{1, 1, 2, 1});
    desc.stride_[0] = 4;  // two pixels followed by two padding bytes
    auto *sample = getOutput()->create(device::getDefaultHostDevice(), desc);
    std::memset(sample->getData(), image->at<uint8_t>(0, 0), 4);
    getOutput()->notifyWritten(sample);
    return base::kStatusCodeOk;
  }
};
REGISTER_NODE("test::StridedSample", StridedSample);

TEST(BatchPreprocessTest, FiveDimensionalBatchPreservesSamplePadding) {
  for (bool planar : {true, false}) {
    dag::Edge input("images"), output("tensor");
    BatchPreprocess node("batch", {&input}, {&output});
    ASSERT_TRUE(node.setNodeKey("test::StridedSample"));
    auto format = planar ? base::kDataFormatNDCHW : base::kDataFormatNDHWC;
    ASSERT_TRUE(node.setDataFormat(format));
    int rows = planar ? 1 : 2;
    std::vector<cv::Mat> images{cv::Mat(rows, 1, CV_8UC1, cv::Scalar(17)),
                                cv::Mat(rows, 1, CV_8UC1, cv::Scalar(29))};
    ASSERT_TRUE(input.set(&images, true));
    ASSERT_TRUE(node.run());
    auto *result = output.getTensor(&node);
    ASSERT_NE(result, nullptr);
    EXPECT_EQ(result->getDataFormat(), format);
    EXPECT_EQ(result->getShape(), planar ? (base::IntVector{1,2,1,1,2})
                                         : (base::IntVector{1,2,1,2,1}));
    EXPECT_EQ(result->getStride(), planar ? (base::SizeVector{8,4,2,2,1})
                                          : (base::SizeVector{8,4,2,1,1}));
    ASSERT_EQ(result->getSize(), 8u);
    auto *bytes = static_cast<uint8_t *>(result->getData());
    for (size_t i = 0; i < 8; ++i) EXPECT_EQ(bytes[i], i < 4 ? 17 : 29);
  }
}
}  // namespace preprocess
}  // namespace nndeploy
