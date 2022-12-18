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
#ifndef _NNDEPLOY_INCLUDE_TASK_NODE_
#define _NNDEPLOY_INCLUDE_TASK_NODE_

#include "nndeploy/include/base/config.h"
#include "nndeploy/include/base/status.h"

#include "nndeploy/include/device/runtime.h"

using namespace nndeploy::base;

namespace nndeploy {
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
}  // namespace nndeploy

#endif