/**
 * @file device.h
 * @author your name (you@domain.com)
 * @brief
 * @version 0.1
 * @date 2022-11-24
 *
 * @copyright Copyright (c) 2022
 *
 */
#ifndef _NN_DEPLOY_TASK_GRAPH_
#define _NN_DEPLOY_TASK_GRAPH_

#include "nn_deploy/base/config.h"
#include "nn_deploy/base/status.h"

#include "nn_deploy/device/runtime.h"

using namespace nn_deploy::base;

namespace nn_deploy {
namespace task {

class graph {
 public:
  explicit Graph(const std::string &name);
  virtual ~Graph();

  virtual Status SetInput(std::vector<std::string> input_name);
  virtual Status SetOutput(std::vector<std::string> input_name);

  virtual Status AddNode(Node *node);

  virtual Status Init();
  virtual Status Deinit();

  virtual Status PreRun();
  virtual Status PostRun();

  virtual Status SetInput(const InputArray& input);
  virtual Value GetOutput(OutputArray& output);

  virtual Status Run();
  virtual Status AsyncRun();

 private:
  std::string name_;
  std::vector<std::string> input_name;
  std::vector<std::string> output_name;
};

}  // namespace device
}  // namespace nn_deploy

#endif