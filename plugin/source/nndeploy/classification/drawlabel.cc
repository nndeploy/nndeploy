#include "nndeploy/classification/drawlabel.h"

#include "nndeploy/base/glic_stl_include.h"
#include "nndeploy/base/time_profiler.h"
#include "nndeploy/dag/node.h"
#include "nndeploy/classification/result.h"
#include "nndeploy/device/device.h"
#include "nndeploy/thread_pool/thread_pool.h"

namespace nndeploy {
namespace classification {

REGISTER_NODE("nndeploy::classification::DrawLable", DrawLable);

}  // namespace classification
}  // namespace nndeploy
