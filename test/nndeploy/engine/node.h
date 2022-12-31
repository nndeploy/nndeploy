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
#ifndef _NNDEPLOY_INCLUDE_ENGINE_NODE_H_
#define _NNDEPLOY_INCLUDE_ENGINE_NODE_H_

#include "nndeploy/include/base/log.h"
#include "nndeploy/include/base/macro.h"
#include "nndeploy/include/base/object.h"
#include "nndeploy/include/base/status.h"
#include "nndeploy/include/base/type.h"
#include "nndeploy/include/device/device.h"
#include "nndeploy/include/engine/config.h"

namespace nndeploy {
namespace engine {

class Node {
 public:
  explicit Node(const std::string& name, std::vector<std::string> input_name,
                std::vector<std::string> output_name);

  virtual ~Node();

  virtual base::Status getConfig(const std::string& key, base::Value& value);

  virtual base::Status setInitConfig(const Config& config);
  virtual base::Status setInitConfig(const std::string& key, const base::Value& value);
  virtual base::Status init();
  virtual base::Status deinit();

  virtual base::Status setPreRunConfig(const Config& config);
  virtual base::Status setPreRunConfig(const std::string& key, const base::Value& config);
  virtual base::Status preRun();
  virtual base::Status postRun();

  virtual base::Status setRunConfig(const Config& config);
  virtual base::Status setRunConfig(const std::string& key, const base::Value& config);
  virtual base::Status setInput(std::string name, const InputArray& input);
  virtual base::Status getOutput(std::string name, OutputArray& output);

  virtual base::Status run();
  virtual base::Status asyncRun();

 private:
  std::string name_;
  std::vector<std::string> input_name_;
  std::vector<std::string> output_name_;
  std::vector<std::string, std::shared_ptr<InputArray>> input_;
  std::vector<std::string, std::shared_ptr<OutputArray>> output_;
};

}  // namespace engine
}  // namespace nndeploy

#endif