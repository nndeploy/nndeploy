
#ifndef _NNDEPLOY_MODEL_STABLE_DIFFUSION_TYPE_H_
#define _NNDEPLOY_MODEL_STABLE_DIFFUSION_TYPE_H_

namespace nndeploy {
namespace model {

enum SchedulerType : int {
  kSchedulerTypeDDIM = 0x0000,
  kSchedulerTypeDPM,
  kSchedulerTypeEulerA,
  kSchedulerTypeLMSD,
  kSchedulerTypePNDM,
  kSchedulerTypeNotSupport,
};

}
}  // namespace nndeploy

#endif /* EF9DBD3E_CD64_43CE_9603_BF0DEBB5E94A */
