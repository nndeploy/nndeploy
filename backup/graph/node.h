
#ifndef _NNDEPLOY_SOURCE_GRAPH_NODE_H_
#define _NNDEPLOY_SOURCE_GRAPH_NODE_H_

#include "nndeploy/source/base/basic.h"
#include "nndeploy/source/base/glic_stl_include.h"
#include "nndeploy/source/base/log.h"
#include "nndeploy/source/base/macro.h"
#include "nndeploy/source/base/object.h"
#include "nndeploy/source/base/status.h"
#include "nndeploy/source/base/string.h"
#include "nndeploy/source/base/value.h"
#include "nndeploy/source/device/buffer.h"
#include "nndeploy/source/device/buffer_pool.h"
#include "nndeploy/source/device/device.h"
#include "nndeploy/source/device/tensor.h"
#include "nndeploy/source/graph/packet.h"

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