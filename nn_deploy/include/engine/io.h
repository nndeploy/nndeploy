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
#ifndef _NN_DEPLOY_TASK_NODE_
#define _NN_DEPLOY_TASK_NODE_

#include "nn_deploy/base/config.h"
#include "nn_deploy/base/status.h"

#include "nn_deploy/device/runtime.h"

using namespace nn_deploy::base;

namespace nn_deploy {
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
}  // namespace nn_deploy

#endif