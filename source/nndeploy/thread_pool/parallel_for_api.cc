
#include "nndeploy/thread_pool/parallel_for_api.h"
#include "nndeploy/thread_pool/parallel_for_api_default.h"

namespace nndeploy {
namespace thread_pool {

std::shared_ptr<ParallelForApi> getParallelForApi(ParallelForApiType type) {
  static std::shared_ptr<ParallelForApi> api =
      std::make_shared<ParallelForApiDefault>();
  return api;
}

}  // namespace thread_pool
}  // namespace nndeploy