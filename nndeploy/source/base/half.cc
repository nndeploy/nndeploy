
#include "nndeploy/source/base/half.h"

#include "nndeploy/source/base/glic_stl_include.h"

namespace nndeploy {
namespace base {

bool convertFromFloatToBfp16(float *fp32, void *bfp16, int count) {
  return false;
}

bool convertFromBfp16ToFloat(void *bfp16, float *fp32, int count) {
  return false;
}

bool convertFromFloatToFp16(float *fp32, void *fp16, int count) {
  return false;
}

bool convertFromFp16ToFloat(void *fp16, float *fp32, int count) {
  return false;
}

}  // namespace base
}  // namespace nndeploy
