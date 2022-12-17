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
#ifndef _NN_DEPLOY_TASK_NODE_
#define _NN_DEPLOY_TASK_NODE_

#include "nn_deploy/base/config.h"
#include "nn_deploy/base/status.h"
#include "nn_deploy/device/runtime.h"


using namespace nn_deploy::base;

namespace nn_deploy {
namespace task {

class Node {
 public:
  explicit Node(const std::string& name, std::vector<std::string> input_name,
                std::vector<std::string> output_name);

  virtual ~Node();

  virtual Status SetInitConfig(const std::string& key, Value& config);
  virtual Status Init();
  virtual Status Deinit();

  virtual Status SetPreRunConfig(const std::string& key, Value& config);
  virtual Status PreRun();
  virtual Status PostRun();

  virtual Status SetRunConfig(const std::string& key, Value& config);
  virtual Status SetInput(const InputArray& input);
  virtual Value GetOutput(OutputArray& output);
  virtual Status Run();
  virtual Status AsyncRun();

  virtual Value GetConfig(const std::string& key);

 private:
  std::string name_;
  std::vector<std::string> input_name;
  std::vector<std::string> output_name;

  std::map<std::string, Value> vulue_map;
};

}  // namespace task
}  // namespace nn_deploy

#endif