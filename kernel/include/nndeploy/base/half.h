
#ifndef _NNDEPLOY_BASE_HALF_H_
#define _NNDEPLOY_BASE_HALF_H_

#include "nndeploy/base/glic_stl_include.h"
#include "nndeploy/base/macro.h"

namespace nndeploy {
namespace base {

typedef union {
  float f;
  uint32_t u;
} cvt_32b;

typedef struct bfp16_struct {
 public:
  uint16_t w = 0;

  bfp16_struct() : w(0) {}

  bfp16_struct(float vf) {
    cvt_32b c;
    c.f = vf;
    w = c.u >> 16;
  }

  operator const float() const {
    cvt_32b c;
    c.u = w << 16;
    return c.f;
  }
} bfp16_t;

extern NNDEPLOY_CC_API bool convertFromFloatToBfp16(float *fp32, void *bfp16,
                                                    int count);

extern NNDEPLOY_CC_API bool convertFromBfp16ToFloat(void *bfp16, float *fp32,
                                                    int count);

extern NNDEPLOY_CC_API bool convertFromFloatToFp16(float *fp32, void *fp16,
                                                   int count);

extern NNDEPLOY_CC_API bool convertFromFp16ToFloat(void *fp16, float *fp32,
                                                   int count);

}  // namespace base
}  // namespace nndeploy

#endif
