
#ifndef _NNDEPLOY_MODEL_STABLE_DIFFUSION_TYPE_H_
#define _NNDEPLOY_MODEL_STABLE_DIFFUSION_TYPE_H_

namespace nndeploy {
namespace stable_diffusion {

enum StableDiffusionType : int {
  kStableDiffusionTypeTextToImage = 0x0000,
  kStableDiffusionTypeImageToImage,
  kStableDiffusionTypeInpaint,
  kStableDiffusionTypeNotSupport,
};

enum SchedulerType : int {
  kSchedulerTypeDDIM = 0x0000,
  kSchedulerTypeDPM,
  kSchedulerTypeEulerA,
  kSchedulerTypeLMSD,
  kSchedulerTypePNDM,
  kSchedulerTypeNotSupport,
};

}  // namespace stable_diffusion
}  // namespace nndeploy

#endif /* _NNDEPLOY_MODEL_STABLE_DIFFUSION_TYPE_H_ */
