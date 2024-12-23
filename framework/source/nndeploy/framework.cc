
#include "nndeploy/framework.h"

#include "nndeploy/base/common.h"
#include "nndeploy/base/glic_stl_include.h"
#include "nndeploy/base/log.h"
#include "nndeploy/base/macro.h"
#include "nndeploy/base/object.h"
#include "nndeploy/base/status.h"
#include "nndeploy/device/device.h"
#include "nndeploy/device/type.h"
#include "nndeploy/device/util.h"

using namespace nndeploy;

int nndeployFrameworkInit() { return 0; }

int nndeployFrameworkDeinit() {
  base::Status status = device::disableDevice();
  if (status != base::kStatusCodeOk) {
    NNDEPLOY_LOGE("nndeployFrameworkDeinit failed\n");
    return -1;
  }
  return 0;
}
