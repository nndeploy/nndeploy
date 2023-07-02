
#ifndef _NNDEPLOY_INCLUDE_GRAPH_NODE_H_
#define _NNDEPLOY_INCLUDE_GRAPH_NODE_H_

#include "nndeploy/include/base/basic.h"
#include "nndeploy/include/base/glic_stl_include.h"
#include "nndeploy/include/base/log.h"
#include "nndeploy/include/base/macro.h"
#include "nndeploy/include/base/object.h"
#include "nndeploy/include/base/status.h"
#include "nndeploy/include/base/string.h"
#include "nndeploy/include/base/value.h"
#include "nndeploy/include/device/buffer.h"
#include "nndeploy/include/device/buffer_pool.h"
#include "nndeploy/include/device/device.h"
#include "nndeploy/include/device/tensor.h"
#include "nndeploy/include/graph/packet.h"

namespace nndeploy {
namespace graph {

class Node {
 public:
  Node(const std::string& name = "", base::Param* param = nullptr,
       Packet* input = nullptr, Packet* output = nullptr);
  virtual ~Node();

  virtual base::Status setName(const std::string& name);
  virtual std::string getName();

  virtual base::Status setParam(base::Param* param);
  virtual base::Param* getParam();

  virtual Packet* getInput();
  virtual Packet* getOutput();

  virtual base::Status setInput(Packet* input);
  virtual base::Status setOutput(Packet* output);

  virtual base::Status init();
  virtual base::Status deinit();

  virtual base::ShapeMap inferOuputShape();

  virtual base::Status run() = 0;

 protected:
  std::string name_;
  base::Param* param_ = nullptr;
  Packet* input_ = nullptr;
  Packet* output_ = nullptr;
};

}  // namespace graph
}  // namespace nndeploy

#endif