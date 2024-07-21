
#include "nndeploy/base/half.h"

#include "nndeploy/base/glic_stl_include.h"
#include "nndeploy/base/half.hpp"
#include "nndeploy/base/log.h"

#if defined(__APPLE__)
#include "TargetConditionals.h"
#if TARGET_OS_IPHONE
#import <Accelerate/Accelerate.h>
#endif
#endif

using namespace half_float;

typedef half fp16_t;
// Largest finite value.
#define HALF_MAX std::numeric_limits<half>::max()
// Smallest positive normal value.
#define HALF_MIN std::numeric_limits<half>::min()
// Smallest finite value.
#define HALF_LOWEST std::numeric_limits<half>::lowest()

namespace nndeploy {
namespace base {

const float MAX_HALF_FLOAT = 65504.0f;
const float MIN_HALF_FLOAT = -65504.0f;

bool convertFromFloatToBfp16(float *fp32, void *bfp16, int count) {
  bfp16_t *bfp16ptr = (bfp16_t *)bfp16;
  for (int i = 0; i < count; ++i) {
    bfp16ptr[i] = fp32[i];
  }

  return true;
}

bool convertFromBfp16ToFloat(void *bfp16, float *fp32, int count) {
  bfp16_t *bfp16ptr = (bfp16_t *)bfp16;
  for (int i = 0; i < count; ++i) {
    fp32[i] = float(bfp16ptr[i]);
  }

  return true;
}

bool convertFromFloatToFp16(float *fp32, void *fp16, int count) {
#if defined(__APPLE__) && TARGET_OS_IPHONE
  vImage_Buffer halfImage, floatImage;
  {
    halfImage.width = count;
    halfImage.height = 1;
    halfImage.rowBytes = count * sizeof(float) / 2;
    halfImage.data = fp16;

    floatImage.width = count;
    floatImage.height = 1;
    floatImage.rowBytes = count * sizeof(float);
    floatImage.data = fp32;
  }

  auto error = vImageConvert_PlanarFtoPlanar16F(&floatImage, &halfImage, 0);
  if (error != kvImageNoError) {
    NNDEPLOY_LOGE("vImageConvert_PlanarFtoPlanar16F error\n");
    return false;
  } else {
    return true;
  }
#else
  bool exceedUplimits = false;
  detail::uint16 *fp16ptr = (detail::uint16 *)fp16;
  for (int i = 0; i < count; ++i) {
    if (fp32[i] > MAX_HALF_FLOAT) {
      exceedUplimits = true;
      NNDEPLOY_LOGE(
          "ERROR: the weights[%d]=%f of conv_layer_data is out of bounds "
          "of float16 max %f. \n",
          i, fp32[i], MAX_HALF_FLOAT);
      fp16ptr[i] =
          detail::float2half<(std::float_round_style)(HALF_ROUND_STYLE)>(
              MAX_HALF_FLOAT);
    } else if (fp32[i] < MIN_HALF_FLOAT) {
      exceedUplimits = true;
      NNDEPLOY_LOGE(
          "ERROR: the weights[%d]=%f of conv_layer_data is out of bounds "
          "of float16 min %f. \n",
          i, fp32[i], MIN_HALF_FLOAT);
      fp16ptr[i] =
          detail::float2half<(std::float_round_style)(HALF_ROUND_STYLE)>(
              MIN_HALF_FLOAT);
      ;
    } else {
      fp16ptr[i] =
          detail::float2half<(std::float_round_style)(HALF_ROUND_STYLE)>(
              fp32[i]);
    }
  }
  return exceedUplimits;
#endif
}

bool convertFromFp16ToFloat(void *fp16, float *fp32, int count) {
#if defined(__APPLE__) && TARGET_OS_IPHONE
  vImage_Buffer halfImage, floatImage;
  {
    halfImage.width = count;
    halfImage.height = 1;
    halfImage.rowBytes = count * sizeof(float) / 2;
    halfImage.data = fp16;

    floatImage.width = count;
    floatImage.height = 1;
    floatImage.rowBytes = count * sizeof(float);
    floatImage.data = fp32;
  }

  auto error = vImageConvert_Planar16FtoPlanarF(&halfImage, &floatImage, 0);
  if (error != kvImageNoError) {
    NNDEPLOY_LOGE("vImageConvert_Planar16FtoPlanarF error\n");
    return false;
  } else {
    return true;
  }
#else
  detail::uint16 *fp16ptr = (detail::uint16 *)fp16;
  for (int i = 0; i < count; ++i) {
    fp32[i] = detail::half2float<float>(fp16ptr[i]);
  }

  return true;
#endif
}

}  // namespace base
}  // namespace nndeploy
