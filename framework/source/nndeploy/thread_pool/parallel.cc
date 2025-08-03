
#include "nndeploy/thread_pool/parallel.h"

#include "nndeploy/thread_pool/parallel_for_api.h"

namespace nndeploy {
namespace thread_pool {

static ParallelForApiType g_parallel_for_api_type = kParallelForApiTypeDefault;

std::string parallelForApiTypeToString(ParallelForApiType type) {
  switch (type) {
    case kParallelForApiTypeDefault:
      return "Default";
    default:
      return "Default";
  }
}

ParallelForApiType stringToParallelForApiType(const std::string &src) {
  if (src == "Default") {
    return kParallelForApiTypeDefault;
  }
  return kParallelForApiTypeDefault;
}

int defaultNumberOfThreads() { return 4; }

void setThreadNum(int num) {
  int num_threads = (num < 0) ? defaultNumberOfThreads() : (unsigned)num;
  std::shared_ptr<ParallelForApi> &api =
      getParallelForApi(g_parallel_for_api_type);
  if (api) {
    api->setThreadNum(num);
  }
}

int getThreadNum() {
  std::shared_ptr<ParallelForApi> &api =
      getParallelForApi(g_parallel_for_api_type);
  if (api) {
    return api->getThreadNum();
  }
  return 0;
}

void parallelFor(const base::Range &range, const ParallelLoopBody &body,
                 double nstripes) {
  if (range.empty()) {
    return;
  }
  std::shared_ptr<ParallelForApi> &api =
      getParallelForApi(g_parallel_for_api_type);
  if (api) {
    api->parallelFor(range, body, nstripes);
  }
  return;
}

}  // namespace thread_pool
}  // namespace nndeploy
