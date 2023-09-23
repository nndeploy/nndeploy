
#include "nndeploy/thread_pool/parallel_for_api.h"

namespace nndeploy {
namespace thread_pool {

std::shared_ptr<ParallelForApi> getParallelForApi() {
  static std::shared_ptr<ParallelForApi> api =
      std::make_shared<ParallelForApiDefault>();
  return api;
}

}  // namespace thread_pool
}  // namespace nndeploy