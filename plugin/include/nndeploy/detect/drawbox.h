#ifndef _NNDEPLOY_DETECT_DRAWBOX_H_
#define _NNDEPLOY_DETECT_DRAWBOX_H_

#include "nndeploy/base/glic_stl_include.h"
#include "nndeploy/base/time_profiler.h"
#include "nndeploy/dag/node.h"
#include "nndeploy/detect/result.h"
#include "nndeploy/device/device.h"
#include "nndeploy/thread_pool/thread_pool.h"

namespace nndeploy {
namespace detect {

class DrawBoxNode : public dag::Node {
 public:
  DrawBoxNode(const std::string &name,
              std::initializer_list<dag::Edge *> inputs,
              std::initializer_list<dag::Edge *> outputs);
  virtual ~DrawBoxNode();

  virtual base::Status run();
};

class YoloMultiConvDrawBoxNode : public dag::Node {
 public:
  YoloMultiConvDrawBoxNode(const std::string &name,
                           std::initializer_list<dag::Edge *> inputs,
                           std::initializer_list<dag::Edge *> outputs);
  virtual ~YoloMultiConvDrawBoxNode();

  virtual base::Status run();
};

}  // namespace detect
}  // namespace nndeploy

#endif