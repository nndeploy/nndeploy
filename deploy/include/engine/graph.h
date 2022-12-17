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
#ifndef _NNKIT_TASK_GRAPH_
#define _NNKIT_TASK_GRAPH_

#include "nnkit/base/config.h"
#include "nnkit/base/status.h"

#include "nnkit/device/runtime.h"

using namespace nnkit::base;

namespace nnkit {
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
}  // namespace nnkit

#endif