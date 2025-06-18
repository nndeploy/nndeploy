#include "nndeploy/segment/drawmask.h"

#include "nndeploy/base/glic_stl_include.h"
#include "nndeploy/base/time_profiler.h"
#include "nndeploy/dag/node.h"
#include "nndeploy/detect/result.h"
#include "nndeploy/device/device.h"
#include "nndeploy/thread_pool/thread_pool.h"

namespace nndeploy {
namespace segment {

REGISTER_NODE("nndeploy::segment::DrawMask", DrawMask);

}  // namespace segment
}  // namespace nndeploy
