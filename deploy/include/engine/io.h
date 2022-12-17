/**
 * @file device.h
 * @author your name (you@domain.com)
 * @brief
 * @version 0.1
 * @date 2022-11-24
 *
 * @copyright Copyright (c) 2022
 * @note ref opencv
 */
#ifndef _NNKIT_TASK_NODE_
#define _NNKIT_TASK_NODE_

#include "nnkit/base/config.h"
#include "nnkit/base/status.h"

#include "nnkit/device/runtime.h"

using namespace nnkit::base;

namespace nnkit {
namespace task {

class InputArray {
 public:
 private:
  std::string name_;
};

class OutputArray {
 public:
 private:
  std::string name_;
};

class InputOutputArray {
 public:
 private:
  Mat mat;
  Tensor tensor;
  std::string name_;
};

}  // namespace task
}  // namespace nnkit

#endif