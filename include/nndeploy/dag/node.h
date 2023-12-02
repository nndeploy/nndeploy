
#ifndef _NNDEPLOY_DAG_NODE_H_
#define _NNDEPLOY_DAG_NODE_H_

#include "nndeploy/base/common.h"
#include "nndeploy/base/glic_stl_include.h"
#include "nndeploy/base/log.h"
#include "nndeploy/base/macro.h"
#include "nndeploy/base/object.h"
#include "nndeploy/base/status.h"
#include "nndeploy/base/string.h"
#include "nndeploy/base/value.h"
#include "nndeploy/dag/edge.h"
#include "nndeploy/device/buffer.h"
#include "nndeploy/device/buffer_pool.h"
#include "nndeploy/device/device.h"
#include "nndeploy/device/tensor.h"

namespace nndeploy {
namespace dag {

/**
 * @brief
 * @note 每个node都要负责分配其输出的edge的内存
 */
class NNDEPLOY_CC_API Node {
 public:
  Node(const std::string& name, Edge* input, Edge* output);
  Node(const std::string& name, std::initializer_list<Edge*> inputs,
       std::initializer_list<Edge*> outputs);

  virtual ~Node();

  std::string getName();

  base::Status setDeviceType(base::DeviceType device_type);
  base::DeviceType getDeviceType();

  virtual base::Status setParam(base::Param* param);
  virtual base::Param* getParam();

  Edge* getInput(int index = 0);
  Edge* getOutput(int index = 0);

  std::vector<Edge*> getAllInput();
  std::vector<Edge*> getAllOutput();

  bool getConstructed();
  bool getInitialized();

  bool isRunning();

  virtual base::Status init();
  virtual base::Status deinit();

  virtual base::Status run() = 0;

 protected:
  std::string name_;
  base::DeviceType device_type_;
  std::shared_ptr<base::Param> param_;
  std::vector<Edge*> inputs_;
  std::vector<Edge*> outputs_;

  bool constructed_ = false;
  bool initialized_ = false;
  bool is_running_ = false;
};

using SingleIONodeFunc =
    std::function<base::Status(Edge* input, Edge* output, base::Param* param)>;

using MultiIONodeFunc = std::function<base::Status(
    std::initializer_list<Edge*> input, std::initializer_list<Edge*> output,
    base::Param* param)>;

}  // namespace dag
}  // namespace nndeploy

#endif
