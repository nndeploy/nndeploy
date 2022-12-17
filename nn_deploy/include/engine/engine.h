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
#ifndef _NN_DEPLOY_TASK_ENGINE_
#define _NN_DEPLOY_TASK_ENGINE_

#include "nn_deploy/base/config.h"
#include "nn_deploy/base/status.h"
#include "nn_deploy/device/runtime.h"


using namespace nn_deploy::base;

namespace nn_deploy {
namespace task {

class Engine {
 public:
  explicit Engine();
  virtual ~Engine();

  virtual Status AddNode(Node *node);
  virtual Status GetNode(Node *node);

 private:
  std::vector<NodePool> node_pool;  // -- 有资源分配问题 和 可重入的
  std::vector<Runtime> runtimes;    // 设备以及内存资源
};

}  // namespace task
}  // namespace nn_deploy

#endif