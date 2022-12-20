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
#ifndef _NNDEPLOY_INCLUDE_TASK_ENGINE_
#define _NNDEPLOY_INCLUDE_TASK_ENGINE_

#include "nndeploy/include/base/config.h"
#include "nndeploy/include/base/status.h"
#include "nndeploy/include/device/runtime.h"


using namespace nndeploy::base;

namespace nndeploy {
namespace task {

class Engine {
 public:
  explicit Engine();
  virtual ~Engine();

  virtual Status AddDevice(const std::string &key, std::shared_ptr<Device> node);
  virtual std::shared_ptr<Device> GetDevice(const std::string &key);

  virtual Status AddNode(const std::string &key, std::shared_ptr<Node> node);
  virtual Status GetNode(const std::string &key, std::shared_ptr<Node> node);

 private:
  std::map<std::string, std::shared_ptr<Node>> node_pool;  // -- 有资源分配问题 和 可重入的
  std::map<std::string, std::shared_ptr<Device>> device_map_;    // 设备以及内存资源
};

}  // namespace task
}  // namespace nndeploy

#endif