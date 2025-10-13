#ifndef _NNDEPLOY_BASE_RING_QUEUE_H_
#define _NNDEPLOY_BASE_RING_QUEUE_H_

#include <cstddef>
#include <utility>
#include <vector>

namespace nndeploy {
namespace base {

template <typename T>
class RingQueue {
 public:
  RingQueue() = default;
  ~RingQueue() = default;

  void reserve(size_t min_capacity) {
    if (min_capacity == 0 || capacity_ >= min_capacity) {
      return;
    }
    size_t new_capacity = nextPowerOfTwo(min_capacity);
    std::vector<T> new_data(new_capacity);
    if (capacity_ != 0) {
      for (size_t i = 0; i < size_; ++i) {
        new_data[i] = std::move(data_[(head_ + i) & mask_]);
      }
    }
    data_.swap(new_data);
    head_ = 0;
    capacity_ = new_capacity;
    mask_ = capacity_ - 1;
  }

  void pushBack(T value) {
    reserve(size_ + 1);
    if (capacity_ == 0) {
      return;
    }
    size_t tail = (head_ + size_) & mask_;
    data_[tail] = std::move(value);
    ++size_;
  }

  T popFront() {
    if (size_ == 0 || capacity_ == 0) {
      return T{};
    }
    T value = std::move(data_[head_]);
    // data_[head_] = T{};
    head_ = (head_ + 1) & mask_;
    --size_;
    return value;
  }

  T front() const {
    if (size_ == 0 || capacity_ == 0) {
      return T{};
    }
    return data_[head_];
  }

  T back() const {
    if (size_ == 0 || capacity_ == 0) {
      return T{};
    }
    size_t tail = (head_ + size_ - 1) & mask_;
    return data_[tail];
  }

  T at(size_t index) const {
    if (index >= size_ || capacity_ == 0) {
      return T{};
    }
    size_t real_index = (head_ + index) & mask_;
    return data_[real_index];
  }

  size_t size() const { return size_; }

  bool empty() const { return size_ == 0; }

  void clear() {
    if (capacity_ == 0) {
      return;
    }
    for (size_t i = 0; i < size_; ++i) {
      data_[(head_ + i) & mask_] = T{};
    }
    head_ = 0;
    size_ = 0;
  }

 private:
  static size_t nextPowerOfTwo(size_t value) {
    size_t capacity = 1;
    while (capacity < value) {
      capacity <<= 1;
    }
    return capacity;
  }

 private:
  std::vector<T> data_;
  size_t head_ = 0;
  size_t size_ = 0;
  size_t capacity_ = 0;
  size_t mask_ = 0;
};

}  // namespace base
}  // namespace nndeploy

#endif
