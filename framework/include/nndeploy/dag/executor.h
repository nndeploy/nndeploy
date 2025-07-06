
#ifndef _NNDEPLOY_DAG_EXECUTOR_H_
#define _NNDEPLOY_DAG_EXECUTOR_H_

#include "nndeploy/base/any.h"
#include "nndeploy/base/common.h"
#include "nndeploy/base/glic_stl_include.h"
#include "nndeploy/base/log.h"
#include "nndeploy/base/macro.h"
#include "nndeploy/base/object.h"
#include "nndeploy/base/status.h"
#include "nndeploy/base/string.h"
#include "nndeploy/dag/edge.h"
#include "nndeploy/dag/node.h"
#include "nndeploy/dag/util.h"

namespace nndeploy {
namespace dag {

class NNDEPLOY_CC_API Executor : public base::NonCopyable {
 public:
  Executor() {};
  virtual ~Executor() {
    if (!is_external_stream_ && stream_ != nullptr) {
      device::destroyStream(stream_);
      stream_ = nullptr;
    }
  };

  void setStream(device::Stream *stream) {
    if (stream_ != nullptr) {
      device::destroyStream(stream_);
    }
    stream_ = stream;
    is_external_stream_ = true;
  }
  device::Stream *getStream() { return stream_; }

  virtual base::Status init(std::vector<EdgeWrapper *> &edge_repository,
                            std::vector<NodeWrapper *> &node_repository) = 0;
  virtual base::Status deinit() = 0;

  virtual base::Status run() = 0;
  virtual bool synchronize() {
    return true;
  }

 protected:
  // must be set by user
  bool is_external_stream_ = false;
  device::Stream *stream_ = nullptr;
};

}  // namespace dag
}  // namespace nndeploy

#endif