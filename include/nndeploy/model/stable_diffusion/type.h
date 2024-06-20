
#ifndef _NNDEPLOY_MODEL_STABLE_DIFFUSION_TYPE_H_
#define _NNDEPLOY_MODEL_STABLE_DIFFUSION_TYPE_H_

namespace nndeploy {
namespace model {

enum StableDiffusionType : int {
  kStableDiffusionTypeTextToImage = 0x0000,
  kStableDiffusionTypeImageToImage,
  kStableDiffusionTypeInpaint,
  kSchedulerTypeNotSupport,
};

enum SchedulerType : int {
  kSchedulerTypeDDIM = 0x0000,
  kSchedulerTypeDPM,
  kSchedulerTypeEulerA,
  kSchedulerTypeLMSD,
  kSchedulerTypePNDM,
  kSchedulerTypeNotSupport,
};

}  // namespace model
}  // namespace nndeploy

#endif /* _NNDEPLOY_MODEL_STABLE_DIFFUSION_TYPE_H_ */
