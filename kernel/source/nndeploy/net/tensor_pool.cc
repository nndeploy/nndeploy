

#include "nndeploy/net/tensor_pool.h"

namespace nndeploy {
namespace net {

TensorPool::TensorPool(device::Device *device,
                       std::vector<TensorWrapper *> &tensor_repository,
                       std::vector<OpWrapper *> &op_repository)
    : device_(device),
      tensor_repository_(tensor_repository),
      op_repository(op_repository) {}

TensorPool::~TensorPool() {}

std::vector<int> getOpOrderIndex(std::vector<OpWrapper *> &producers,
                                 std::vector<OpWrapper *> &consumers,
                                 std::vector<OpWrapper *> &op_repository) {
  std::vector<int> order_index;

  for (size_t i = 0; i < op_repository.size(); i++) {
    for (size_t j = 0; j < producers.size(); j++) {
      if (op_repository[i] == producers[j]) {
        order_index.push_back(i);
        break;
      }
    }

    for (size_t j = 0; j < consumers.size(); j++) {
      if (op_repository[i] == consumers[j]) {
        order_index.push_back(i);
        break;
      }
    }
  }

  return order_index;
}

bool isInterval(std::array<int, 2> &interval,
                std::vector<std::array<int, 2>> &intervals) {
  for (size_t i = 0; i < intervals.size(); i++) {
    if (interval[0] >= intervals[i][0] && interval[0] <= intervals[i][1]) {
      return true;
    }
    if (interval[1] >= intervals[i][0] && interval[1] <= intervals[i][1]) {
      return true;
    }
  }
  return true;
}

}  // namespace net
}  // namespace nndeploy
