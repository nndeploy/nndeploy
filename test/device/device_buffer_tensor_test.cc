#include "nndeploy/device/buffer.h"
#include "nndeploy/device/device.h"
#include "nndeploy/device/tensor.h"

#include <gtest/gtest.h>

#include <cstdlib>
#include <cstring>
#include <unordered_set>

namespace {

class CountingDevice : public nndeploy::device::Device {
 public:
  CountingDevice()
      : Device(nndeploy::base::DeviceType(nndeploy::base::kDeviceTypeCodeCpu,
                                          0)) {}

  nndeploy::device::BufferDesc toBufferDesc(
      const nndeploy::device::TensorDesc &desc,
      const nndeploy::base::IntVector &config) override {
    size_t size = desc.data_type_.size();
    if (desc.stride_.empty()) {
      for (int dim : desc.shape_) {
        size *= dim;
      }
    } else {
      size = desc.stride_[0] * desc.shape_[0];
    }
    return nndeploy::device::BufferDesc(size, config);
  }

  void *allocate(size_t size) override {
    void *ptr = std::malloc(size == 0 ? 1 : size);
    EXPECT_NE(ptr, nullptr);
    if (ptr != nullptr) {
      ++allocate_count_;
      live_allocations_.insert(ptr);
    }
    return ptr;
  }

  void *allocate(const nndeploy::device::BufferDesc &desc) override {
    return allocate(desc.getRealSize());
  }

  void deallocate(void *ptr) override {
    if (ptr == nullptr) {
      return;
    }
    auto iter = live_allocations_.find(ptr);
    EXPECT_NE(iter, live_allocations_.end());
    if (iter != live_allocations_.end()) {
      live_allocations_.erase(iter);
    }
    ++deallocate_count_;
    std::free(ptr);
  }

  nndeploy::base::Status copy(void *src, void *dst, size_t size,
                              nndeploy::device::Stream *stream =
                                  nullptr) override {
    if (src == nullptr || dst == nullptr) {
      return nndeploy::base::kStatusCodeErrorNullParam;
    }
    std::memcpy(dst, src, size);
    return nndeploy::base::kStatusCodeOk;
  }

  nndeploy::base::Status download(void *src, void *dst, size_t size,
                                  nndeploy::device::Stream *stream =
                                      nullptr) override {
    return copy(src, dst, size, stream);
  }

  nndeploy::base::Status upload(void *src, void *dst, size_t size,
                                nndeploy::device::Stream *stream =
                                    nullptr) override {
    return copy(src, dst, size, stream);
  }

  nndeploy::base::Status copy(nndeploy::device::Buffer *src,
                              nndeploy::device::Buffer *dst,
                              nndeploy::device::Stream *stream =
                                  nullptr) override {
    return copy(src->getData(), dst->getData(),
                std::min(src->getRealSize(), dst->getRealSize()), stream);
  }

  nndeploy::base::Status download(nndeploy::device::Buffer *src,
                                  nndeploy::device::Buffer *dst,
                                  nndeploy::device::Stream *stream =
                                      nullptr) override {
    return copy(src, dst, stream);
  }

  nndeploy::base::Status upload(nndeploy::device::Buffer *src,
                                nndeploy::device::Buffer *dst,
                                nndeploy::device::Stream *stream =
                                    nullptr) override {
    return copy(src, dst, stream);
  }

  nndeploy::base::Status init() override {
    return nndeploy::base::kStatusCodeOk;
  }

  nndeploy::base::Status deinit() override {
    return nndeploy::base::kStatusCodeOk;
  }

  int allocateCount() const { return allocate_count_; }
  int deallocateCount() const { return deallocate_count_; }
  size_t liveAllocationCount() const { return live_allocations_.size(); }

 private:
  int allocate_count_ = 0;
  int deallocate_count_ = 0;
  std::unordered_set<void *> live_allocations_;
};

class BufferTensorLifetimeTest : public testing::Test {
 protected:
  CountingDevice device_;
};

TEST_F(BufferTensorLifetimeTest, ExternalOwnerFollowsCopiesAndMoves) {
  using namespace nndeploy::device;
  auto data = std::make_shared<int>(42);
  std::weak_ptr<int> lifetime = data;
  {
    Buffer survivor(&device_, 1);
    {
      Buffer original(&device_, BufferDesc(sizeof(int)), data);
      data.reset();
      Buffer copy(original);
      survivor = std::move(copy);
    }
    ASSERT_FALSE(lifetime.expired());
    EXPECT_EQ(*static_cast<int *>(survivor.getData()), 42);
  }
  EXPECT_TRUE(lifetime.expired());
  EXPECT_EQ(device_.deallocateCount(), 1);  // Only survivor's original allocation.
}

TEST_F(BufferTensorLifetimeTest, BufferCopyAssignmentReleasesOldStorage) {
  {
    nndeploy::device::Buffer lhs(&device_, 16);
    nndeploy::device::Buffer rhs(&device_, 32);

    EXPECT_EQ(device_.allocateCount(), 2);
    EXPECT_EQ(device_.deallocateCount(), 0);

    lhs = rhs;

    EXPECT_EQ(device_.deallocateCount(), 1);
    EXPECT_EQ(device_.liveAllocationCount(), 1U);
    EXPECT_EQ(lhs.getData(), rhs.getData());
  }

  EXPECT_EQ(device_.allocateCount(), 2);
  EXPECT_EQ(device_.deallocateCount(), 2);
  EXPECT_EQ(device_.liveAllocationCount(), 0U);
}

TEST_F(BufferTensorLifetimeTest, BufferMoveAssignmentReleasesOldStorage) {
  void *moved_data = nullptr;

  {
    nndeploy::device::Buffer lhs(&device_, 16);
    nndeploy::device::Buffer rhs(&device_, 32);
    moved_data = rhs.getData();

    lhs = std::move(rhs);

    EXPECT_EQ(device_.deallocateCount(), 1);
    EXPECT_EQ(device_.liveAllocationCount(), 1U);
    EXPECT_EQ(lhs.getData(), moved_data);
    EXPECT_EQ(rhs.getData(), nullptr);
  }

  EXPECT_EQ(device_.allocateCount(), 2);
  EXPECT_EQ(device_.deallocateCount(), 2);
  EXPECT_EQ(device_.liveAllocationCount(), 0U);
}

TEST_F(BufferTensorLifetimeTest, BufferSharedCopiesReleaseStorageOnce) {
  {
    nndeploy::device::Buffer original(&device_, 64);
    {
      nndeploy::device::Buffer copy = original;
      EXPECT_EQ(copy.getData(), original.getData());
      EXPECT_EQ(device_.deallocateCount(), 0);
    }
    EXPECT_EQ(device_.deallocateCount(), 0);
  }

  EXPECT_EQ(device_.allocateCount(), 1);
  EXPECT_EQ(device_.deallocateCount(), 1);
  EXPECT_EQ(device_.liveAllocationCount(), 0U);
}

TEST_F(BufferTensorLifetimeTest, TensorCopyAssignmentReleasesOldStorage) {
  nndeploy::device::TensorDesc desc(nndeploy::base::dataTypeOf<float>(),
                                    nndeploy::base::kDataFormatN,
                                    {4});

  {
    nndeploy::device::Tensor lhs(&device_, desc, "lhs");
    nndeploy::device::Tensor rhs(&device_, desc, "rhs");

    EXPECT_EQ(device_.allocateCount(), 2);
    EXPECT_EQ(device_.deallocateCount(), 0);

    lhs = rhs;

    EXPECT_EQ(device_.deallocateCount(), 1);
    EXPECT_EQ(device_.liveAllocationCount(), 1U);
    ASSERT_NE(lhs.getBuffer(), nullptr);
    ASSERT_NE(rhs.getBuffer(), nullptr);
    EXPECT_EQ(lhs.getBuffer()->getData(), rhs.getBuffer()->getData());
  }

  EXPECT_EQ(device_.allocateCount(), 2);
  EXPECT_EQ(device_.deallocateCount(), 2);
  EXPECT_EQ(device_.liveAllocationCount(), 0U);
}

TEST_F(BufferTensorLifetimeTest, TensorMoveAssignmentReleasesOldStorage) {
  nndeploy::device::TensorDesc desc(nndeploy::base::dataTypeOf<float>(),
                                    nndeploy::base::kDataFormatN,
                                    {4});
  void *moved_data = nullptr;

  {
    nndeploy::device::Tensor lhs(&device_, desc, "lhs");
    nndeploy::device::Tensor rhs(&device_, desc, "rhs");
    moved_data = rhs.getBuffer()->getData();

    lhs = std::move(rhs);

    EXPECT_EQ(device_.deallocateCount(), 1);
    EXPECT_EQ(device_.liveAllocationCount(), 1U);
    ASSERT_NE(lhs.getBuffer(), nullptr);
    EXPECT_EQ(lhs.getBuffer()->getData(), moved_data);
    EXPECT_EQ(rhs.getBuffer(), nullptr);
  }

  EXPECT_EQ(device_.allocateCount(), 2);
  EXPECT_EQ(device_.deallocateCount(), 2);
  EXPECT_EQ(device_.liveAllocationCount(), 0U);
}

TEST_F(BufferTensorLifetimeTest, TensorSharedDeallocateReleasesStorageOnce) {
  nndeploy::device::TensorDesc desc(nndeploy::base::dataTypeOf<float>(),
                                    nndeploy::base::kDataFormatN,
                                    {8});

  nndeploy::device::Tensor original(&device_, desc, "origin");
  nndeploy::device::Tensor copy = original;

  original.deallocate();
  EXPECT_EQ(device_.deallocateCount(), 0);
  EXPECT_EQ(device_.liveAllocationCount(), 1U);

  copy.deallocate();
  EXPECT_EQ(device_.deallocateCount(), 1);
  EXPECT_EQ(device_.liveAllocationCount(), 0U);
}

TEST_F(BufferTensorLifetimeTest, BufferRoundTripAndTruncatedInput) {
  const unsigned char bytes[] = {0xff, 0, 0x81, 42};
  nndeploy::device::Buffer original(&device_, sizeof(bytes));
  std::memcpy(original.getData(), bytes, sizeof(bytes));
  std::string wire;
  ASSERT_TRUE(original.serialize(wire));
  ASSERT_EQ(sizeof(uint64_t) + sizeof(bytes), wire.size());

  void *ptr = original.getData();
  for (size_t len = 0; len < wire.size(); ++len) {
    ASSERT_FALSE(original.deserialize(wire.substr(0, len)));
    ASSERT_EQ(ptr, original.getData());
    ASSERT_EQ(&device_, original.getDevice());
    ASSERT_EQ(sizeof(bytes), original.getSize());
    ASSERT_EQ(0, std::memcmp(bytes, ptr, sizeof(bytes)));
  }
  nndeploy::device::Buffer restored(&device_, 1);
  ASSERT_TRUE(restored.deserialize(wire));
  ASSERT_EQ(sizeof(bytes), restored.getSize());
  EXPECT_EQ(0, std::memcmp(bytes, restored.getData(), sizeof(bytes)));
}

TEST_F(BufferTensorLifetimeTest, TensorRoundTripPreservesDescriptorAndBytes) {
  nndeploy::device::TensorDesc desc(nndeploy::base::dataTypeOf<uint8_t>(),
                                    nndeploy::base::kDataFormatNC, {2, 3}, {3, 1});
  nndeploy::device::Tensor original(&device_, desc, "features");
  const unsigned char bytes[] = {0, 17, 255, 1, 128, 9};
  std::memcpy(original.getData(), bytes, sizeof(bytes));
  std::string wire;
  ASSERT_TRUE(original.serialize(wire));
  nndeploy::device::Tensor restored;
  ASSERT_TRUE(restored.deserialize(wire));
  EXPECT_EQ(original.getName(), restored.getName());
  EXPECT_EQ(desc.data_type_, restored.getDesc().data_type_);
  EXPECT_EQ(desc.data_format_, restored.getDesc().data_format_);
  EXPECT_EQ(desc.shape_, restored.getDesc().shape_);
  EXPECT_EQ(desc.stride_, restored.getDesc().stride_);
  ASSERT_EQ(sizeof(bytes), restored.getBuffer()->getRealSize());
  EXPECT_EQ(0, std::memcmp(bytes, restored.getData(), sizeof(bytes)));
}

TEST(TensorLayout, ChecksByteStridesWithoutReadingTensorData) {
  using namespace nndeploy;
  device::TensorDesc desc(base::dataTypeOf<float>(), base::kDataFormatNCHW,
                          {1, 3, 2, 5});
  EXPECT_TRUE(device::Tensor(desc).isContinue());
  desc.stride_[0] = 0;
  EXPECT_TRUE(device::Tensor(desc).isContinue());
  desc.stride_[2] += sizeof(float);
  EXPECT_FALSE(device::Tensor(desc).isContinue());
  desc.stride_.push_back(1);
  EXPECT_FALSE(device::Tensor(desc).isContinue());
  desc.stride_.clear();
  EXPECT_TRUE(device::Tensor(desc).isContinue());
  desc.shape_[1] = -1;
  EXPECT_FALSE(device::Tensor(desc).isContinue());
  desc.shape_[1] = 0;
  EXPECT_TRUE(device::Tensor(desc).isContinue());
}

}  // namespace
