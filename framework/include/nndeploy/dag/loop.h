#ifndef _NNDEPLOY_DAG_LOOP_H_
#define _NNDEPLOY_DAG_LOOP_H_

#include "nndeploy/base/any.h"
#include "nndeploy/base/common.h"
#include "nndeploy/base/glic_stl_include.h"
#include "nndeploy/base/log.h"
#include "nndeploy/base/macro.h"
#include "nndeploy/base/object.h"
#include "nndeploy/base/status.h"
#include "nndeploy/base/string.h"
#include "nndeploy/dag/edge.h"
#include "nndeploy/dag/graph.h"
#include "nndeploy/dag/node.h"
#include "nndeploy/dag/util.h"
#include "nndeploy/device/buffer.h"
#include "nndeploy/device/device.h"
#include "nndeploy/device/memory_pool.h"
#include "nndeploy/device/tensor.h"

namespace nndeploy {
namespace dag {

class NNDEPLOY_CC_API Loop : public Graph {
 public:
  Loop(const std::string& name);
  Loop(const std::string& name, std::vector<dag::Edge*> inputs,
       std::vector<dag::Edge*> outputs);
  virtual ~Loop();

  virtual base::Status initStart() { return base::kStatusCodeOk; }
  virtual base::Status initEnd() { return base::kStatusCodeOk; }
  virtual base::Status init();
  virtual base::Status deinit();

  virtual int loops() = 0;
  virtual base::Status iterBefore() { return base::kStatusCodeOk; }
  virtual base::Status iterAfter() { return base::kStatusCodeOk; }
  virtual base::Status run();

 protected:
  virtual base::Status executor();
};

// 固定次数的loop
class NNDEPLOY_CC_API FixedLoop : public Loop {
 public:
  FixedLoop(const std::string& name);
  FixedLoop(const std::string& name, std::vector<dag::Edge*> inputs,
            std::vector<dag::Edge*> outputs);
  virtual ~FixedLoop();

  virtual int loops() override;

  virtual void setLoops(int loops);

  using dag::Node::serialize;
  virtual base::Status serialize(
      rapidjson::Value& json,
      rapidjson::Document::AllocatorType& allocator) override;
  using dag::Node::deserialize;
  virtual base::Status deserialize(rapidjson::Value& json) override;

 protected:
  int loops_ = 0;
};

}  // namespace dag
}  // namespace nndeploy

#endif /* _NNDEPLOY_DAG_LOOP_H_ */