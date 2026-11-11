#include <gtest/gtest.h>
#include "nndeploy/segment/result.h"

TEST(SegMaskLifetime, LastResultReleasesMask) {
  using namespace nndeploy;
  std::weak_ptr<device::Tensor> weak;
  {
    segment::SegMaskResult first;
    segment::SegMaskItem item;
    item.mask_ = std::make_shared<device::Tensor>("kept");
    weak = item.mask_;
    first.masks_.push_back(std::move(item));
    auto second = first;
    first.masks_.clear();
    ASSERT_FALSE(weak.expired());
    EXPECT_EQ(second.masks_[0].mask_->getName(), "kept");
  }
  EXPECT_TRUE(weak.expired());
}
