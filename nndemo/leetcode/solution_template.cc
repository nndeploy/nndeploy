/**
 * @brief
 *
 */
#include <cassert>
#include <cstdio>
#include <cstdlib>

/**
 * @brief
 *
 */
#include <ostream>
#include <sstream>
#include <string>

/**
 * @brief
 *
 */
#include <memory>

/**
 * @brief
 *
 */
#include <array>
#include <list>
#include <map>
#include <queue>
#include <set>
#include <unordered_map>
#include <unordered_set>
#include <vector>

/**
 * @brief
 *
 */
#include <algorithm>
#include <functional>
#include <iterator>

/**
 * @brief
 *
 */
#include <condition_variable>
#include <future>
#include <mutex>
#include <thread>

/**
 * @brief
 *
 */
#include <stdexcept>

namespace nndemo {
namespace leetcode{

class Solution {
 public:
  static int RemoveDuplicateFromSortedArrayV0(std::vector<int> &nums) {
    int index = 0;
    for (int i = 0; i < nums.size(); ++i) {
      if (nums[index] != nums[i]) {
        nums[++index] = nums[i];
      }
    }
    return index + 1;
  }

  static int RemoveDuplicateFromSortedArrayV1(std::vector<int> &nums) {
    return std::distance(nums.begin(), std::unique(nums.begin(), nums.end()));
  }

  // static int RemoveDuplicateFromSortedArrayV3(std::vector<int> &nums) {
  //   return std::distance(nums.begin(), Solution::RemoveDuplicate(nums.begin(), nums.end(), nums.begin()));
  // }

  template<typename T>
  static int RemoveDuplicate(T first, T last, T output) {
    while(first != last) {
      *output++ = *first;
      first = std::upper_bound(first, last, *first);
    }
  }
};

}
}

int main() {
  return 0;
}

